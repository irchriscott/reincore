//! Bounded, syntax-aware source reference recognition for the privacy boundary.
//!
//! Only exact AST expression locations are exempted, never a spelling everywhere
//! in a document. Strings/comments and opaque data files cannot establish source
//! bindings. This is syntax recognition, not execution or semantic name resolution.

use std::collections::HashMap;
use std::ops::Range;
use std::time::{Duration, Instant};
use tree_sitter::{Language, Node, ParseOptions, Parser, Tree};

const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_NODES: usize = 50_000;
const MAX_ANALYSIS_TIME: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SourceLanguage {
    Python,
    Javascript,
    Typescript,
    Tsx,
    Rust,
    Go,
    Java,
    Csharp,
    Kotlin,
    C,
    Cpp,
}

impl SourceLanguage {
    fn grammar(self) -> Language {
        match self {
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::Javascript => tree_sitter_javascript::LANGUAGE.into(),
            Self::Typescript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Tsx => tree_sitter_typescript::LANGUAGE_TSX.into(),
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::Csharp => tree_sitter_c_sharp::LANGUAGE.into(),
            Self::Kotlin => tree_sitter_kotlin_ng::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        }
    }

    fn named(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "py" | "python" => Self::Python,
            "js" | "jsx" | "mjs" | "cjs" | "javascript" => Self::Javascript,
            "ts" | "mts" | "cts" | "typescript" => Self::Typescript,
            "tsx" => Self::Tsx,
            "rs" | "rust" => Self::Rust,
            "go" | "golang" => Self::Go,
            "java" => Self::Java,
            "cs" | "csharp" | "c#" => Self::Csharp,
            "kt" | "kts" | "kotlin" => Self::Kotlin,
            "c" | "h" => Self::C,
            "cc" | "cpp" | "cxx" | "hpp" | "hxx" | "c++" => Self::Cpp,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum SourceHint {
    #[default]
    Auto,
    Language(SourceLanguage),
    /// A supplied path/fence identifies data or an unsupported language. Do not
    /// reinterpret it as a different supported language to gain an exemption.
    Opaque,
}

impl SourceHint {
    pub(crate) fn from_path(path: &str) -> Self {
        let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
        let extension = name.rsplit('.').next().unwrap_or("");
        // C++ headers share .h with C. Its grammar also accepts ordinary C
        // declarations; .c remains routed to the dedicated C grammar.
        if extension.eq_ignore_ascii_case("h") {
            return Self::Language(SourceLanguage::Cpp);
        }
        match SourceLanguage::named(extension) {
            Some(language) => Self::Language(language),
            None if matches!(extension.to_ascii_lowercase().as_str(), "md" | "markdown") => {
                Self::Auto
            }
            None => Self::Opaque,
        }
    }
}

#[derive(Default)]
pub(crate) struct SourceReferences {
    expressions: Vec<Range<usize>>,
    definite: Vec<Range<usize>>,
}

impl SourceReferences {
    pub(crate) fn analyze(text: &str, hint: SourceHint) -> Self {
        let mut result = Self::default();
        if hint == SourceHint::Opaque || text.len() > MAX_SOURCE_BYTES {
            return result;
        }
        let started = Instant::now();
        let mut nodes = 0;
        // Fence/path context is local to its own region. Do not let one supported
        // code block exempt identifiers in neighboring data or unsupported code.
        let regions = source_regions(text, hint);
        for (range, hint, patch) in regions {
            if started.elapsed() >= MAX_ANALYSIS_TIME || nodes >= MAX_NODES {
                break;
            }
            if hint == SourceHint::Opaque {
                continue;
            }
            // Borrow the region instead of copying it (up to 256 KB) when no
            // patch normalization is needed.
            let source: std::borrow::Cow<'_, str> = if patch {
                std::borrow::Cow::Owned(normalize_patch(&text[range.clone()]))
            } else {
                std::borrow::Cow::Borrowed(&text[range.clone()])
            };
            let languages = match hint {
                SourceHint::Language(language) => vec![language],
                SourceHint::Auto => candidates(&source),
                SourceHint::Opaque => Vec::new(),
            };
            for language in languages {
                let Some((references, definite, evidence)) = parse_references(
                    &source,
                    language,
                    hint == SourceHint::Auto,
                    started,
                    &mut nodes,
                ) else {
                    continue;
                };
                if hint == SourceHint::Auto && !evidence {
                    continue;
                }
                result.expressions.extend(
                    references
                        .into_iter()
                        .map(|span| span.start + range.start..span.end + range.start),
                );
                result.definite.extend(
                    definite
                        .into_iter()
                        .map(|span| span.start + range.start..span.end + range.start),
                );
                break;
            }
        }
        result.expressions.sort_unstable_by_key(|span| span.start);
        result.definite.sort_unstable_by_key(|span| span.start);
        result
    }

    pub(crate) fn contains(&self, start: usize, end: usize) -> bool {
        contains_range(&self.expressions, start, end)
    }

    pub(crate) fn recognizes(&self, start: usize, end: usize, generated: bool) -> bool {
        if generated {
            contains_range(&self.definite, start, end)
        } else {
            self.contains(start, end)
        }
    }
}

fn contains_range(expressions: &[Range<usize>], start: usize, end: usize) -> bool {
    let upper = expressions.partition_point(|span| span.start <= start);
    expressions[..upper]
        .iter()
        .rev()
        .take_while(|span| span.start == start)
        .any(|span| end <= span.end)
}

fn candidates(source: &str) -> Vec<SourceLanguage> {
    use SourceLanguage::*;
    // Ranking only chooses which grammar to try first. Auto recognition still
    // requires a complete, error-free parse with positive source-code evidence.
    let mut ranked = Vec::new();
    for (marker, language) in [
        ("package ", Go),
        (":=", Go),
        ("func ", Go),
        ("namespace ", Csharp),
        ("using ", Csharp),
        ("fun ", Kotlin),
        ("val ", Kotlin),
        ("fn ", Rust),
        ("json!(", Rust),
        ("let mut ", Rust),
        ("interface ", Typescript),
        ("const ", Typescript),
        ("=>", Typescript),
        ("#include", Cpp),
        ("std::", Cpp),
        ("public class ", Java),
    ] {
        if source.contains(marker) && !ranked.contains(&language) {
            ranked.push(language);
        }
    }
    for language in [
        Python, Typescript, Javascript, Rust, Go, Java, Csharp, Kotlin, C, Cpp, Tsx,
    ] {
        if !ranked.contains(&language) {
            ranked.push(language);
        }
    }
    ranked
}

fn parse_references(
    source: &str,
    language: SourceLanguage,
    auto: bool,
    started: Instant,
    visited: &mut usize,
) -> Option<(Vec<Range<usize>>, Vec<Range<usize>>, bool)> {
    if started.elapsed() >= MAX_ANALYSIS_TIME || *visited >= MAX_NODES {
        return None;
    }
    let mut parser = Parser::new();
    parser.set_language(&language.grammar()).ok()?;
    let unicode_source =
        (language == SourceLanguage::Kotlin).then(|| kotlin_identifier_aliases(source));
    let parser_source = unicode_source.as_deref().unwrap_or(source);
    let mut tree = parse_tree(&mut parser, parser_source, started)?;
    // Partial edits can need an object/function/class wrapper in any language.
    // A synthetic wrapper establishes syntax, NOT source-code evidence. Only
    // declarations/calls inside the original text or explicit instance members
    // can establish that evidence; credential data must not gain an exemption
    // merely because we placed it inside a generated function.
    let mut wrapped_source = None;
    let mut original_start = 0;
    if tree.root_node().has_error() {
        for (prefix, suffix) in fragment_wrappers(language) {
            if started.elapsed() >= MAX_ANALYSIS_TIME {
                return None;
            }
            let candidate = format!("{prefix}{parser_source}{suffix}");
            parser.reset();
            let candidate_tree = parse_tree(&mut parser, &candidate, started)?;
            if !candidate_tree.root_node().has_error() {
                original_start = prefix.len();
                tree = candidate_tree;
                wrapped_source = Some(candidate);
                break;
            }
        }
    }
    if auto && tree.root_node().has_error() {
        return None;
    }
    let analyzed_source = wrapped_source.as_deref().unwrap_or(parser_source);
    let mut stack = vec![tree.root_node()];
    let mut expressions = Vec::new();
    let mut definite = Vec::new();
    let mut instance_members = Vec::new();
    let mut uncertain = Vec::new();
    let mut bindings: HashMap<&str, Vec<(Range<usize>, usize)>> = HashMap::new();
    let mut code_evidence = false;
    while let Some(node) = stack.pop() {
        *visited += 1;
        if *visited > MAX_NODES || started.elapsed() >= MAX_ANALYSIS_TIME {
            // Do not release a partially traversed analysis after exhaustion.
            return None;
        }
        let kind = node.kind();
        if node.is_error() || node.is_missing() || is_literal_or_comment(kind) {
            continue;
        }
        let span = node.byte_range();
        let in_original = span.start >= original_start && span.end <= original_start + source.len();
        let environment_value = auto && is_environment_style_assignment(node, analyzed_source);
        code_evidence |= in_original && is_code_evidence(node);
        if in_original && is_instance_member(node, analyzed_source, language) && !environment_value
        {
            instance_members.push(span.start - original_start..span.end - original_start);
        }
        if matches!(kind, "identifier" | "simple_identifier") {
            if let Some(scope) = binding_scope(node, language) {
                let span = node.byte_range();
                if span.start >= original_start && span.end <= original_start + source.len() {
                    bindings
                        .entry(&source[span.start - original_start..span.end - original_start])
                        .or_default()
                        .push((scope, span.start));
                }
            }
        }
        if !node.has_error() && is_reference_expression(kind) && !environment_value {
            let span = node.byte_range();
            if span.start >= original_start && span.end <= original_start + source.len() {
                let mapped = span.start - original_start..span.end - original_start;
                // A bare identifier is only a definite reference when it is
                // actually bound (declared or imported) in scope, in every
                // mode: an unbound generated-looking name in a known file
                // (`api_key = sk_live_...` written without quotes) must not
                // gain an exemption just because the file type is known.
                if !matches!(kind, "identifier" | "simple_identifier") {
                    definite.push(mapped.clone());
                } else {
                    uncertain.push((mapped.clone(), span.start));
                }
                expressions.push(mapped);
            }
        }
        for index in (0..node.named_child_count()).rev() {
            if let Some(child) = node.named_child(index) {
                stack.push(child);
            }
        }
    }
    if auto && !code_evidence {
        // A bare object member is convincing only for an explicit self/this/
        // cls/super/base receiver. Do not exempt neighboring unbound words in
        // an object fragment, or members rooted in arbitrary credential data.
        let evidence = !instance_members.is_empty();
        return Some((instance_members.clone(), instance_members, evidence));
    }
    for (span, position) in uncertain {
        if started.elapsed() >= MAX_ANALYSIS_TIME {
            return None;
        }
        let name = &source[span.clone()];
        if bindings.get(name).is_some_and(|declarations| {
            declarations
                .iter()
                .any(|(scope, declared)| scope.contains(&position) && *declared <= position)
        }) {
            definite.push(span);
        }
    }
    Some((expressions, definite, code_evidence))
}

fn binding_scope(node: Node<'_>, language: SourceLanguage) -> Option<Range<usize>> {
    // Imported names are bound for the whole file in every supported
    // language; their scope is the root, not the import statement.
    let mut ancestor = node.parent();
    for _ in 0..32 {
        let Some(parent) = ancestor else {
            break;
        };
        if matches!(
            parent.kind(),
            "import_statement"
                | "import_from_statement"
                | "import_declaration"
                | "import_spec"
                | "use_declaration"
                | "using_directive"
                | "import_header"
        ) {
            let mut root = parent;
            while let Some(up) = root.parent() {
                root = up;
            }
            return Some(root.byte_range());
        }
        ancestor = parent.parent();
    }
    let span = node.byte_range();
    let mut ancestor = node.parent();
    let mut binding = false;
    for _ in 0..128 {
        let Some(parent) = ancestor else {
            break;
        };
        let kind = parent.kind();
        let field = match kind {
            "assignment" | "assignment_expression" | "short_var_declaration" => {
                parent.child_by_field_name("left")
            }
            "variable_declarator" | "var_spec" => parent.child_by_field_name("name"),
            "let_declaration" => parent.child_by_field_name("pattern"),
            "function_definition"
            | "function_declaration"
            | "function_item"
            | "method_declaration"
            | "method_definition"
            | "class_declaration"
            | "class_definition"
            | "interface_declaration"
            | "type_declaration"
            | "enum_declaration" => parent.child_by_field_name("name"),
            "parameter"
            | "formal_parameter"
            | "required_parameter"
            | "optional_parameter"
            | "parameter_declaration"
            | "typed_parameter"
            | "default_parameter" => parent
                .child_by_field_name("name")
                .or_else(|| parent.child_by_field_name("pattern"))
                .or_else(|| parent.named_child(0)),
            "variable_declaration" => parent.named_child(0),
            "parameters"
                if node
                    .parent()
                    .is_some_and(|direct| direct.id() == parent.id()) =>
            {
                Some(node)
            }
            _ => None,
        };
        if field
            .is_some_and(|field| field.start_byte() <= span.start && span.end <= field.end_byte())
        {
            binding = true;
            break;
        }
        if kind.contains("function") || kind.contains("statement") || kind == "block" {
            break;
        }
        ancestor = parent.parent();
    }
    if !binding {
        return None;
    }
    let mut ancestor = node.parent();
    let mut scope = node;
    for _ in 0..128 {
        let Some(parent) = ancestor else {
            return Some(scope.byte_range());
        };
        scope = parent;
        let kind = parent.kind();
        if matches!(
            kind,
            "function_definition"
                | "function_declaration"
                | "function_item"
                | "method_declaration"
                | "arrow_function"
                | "lambda_expression"
        ) || (language != SourceLanguage::Python
            && matches!(kind, "block" | "statement_block" | "compound_statement"))
        {
            return Some(scope.byte_range());
        }
        ancestor = parent.parent();
    }
    None
}

fn parse_tree(parser: &mut Parser, source: &str, started: Instant) -> Option<Tree> {
    let mut cancel = |_: &tree_sitter::ParseState| started.elapsed() >= MAX_ANALYSIS_TIME;
    let mut input =
        |offset: usize, _: tree_sitter::Point| source.as_bytes().get(offset..).unwrap_or_default();
    parser.parse_with_options(
        &mut input,
        None,
        Some(ParseOptions::new().progress_callback(&mut cancel)),
    )
}

fn fragment_wrappers(language: SourceLanguage) -> &'static [(&'static str, &'static str)] {
    use SourceLanguage::*;
    match language {
        Python => &[("{\n", "\n}")],
        Javascript | Typescript | Tsx => &[
            ("function privacyFragment(){\n", "\n}"),
            ("const privacyFragment = {\n", "\n};"),
        ],
        Rust => &[("fn privacy_fragment(){\n", "\n}")],
        Go => &[("package privacy\nfunc privacyFragment(){\n", "\n}")],
        Java => &[("class PrivacyFragment { void fragment(){\n", "\n}}")],
        Csharp => &[("class PrivacyFragment { void Fragment(){\n", "\n}}")],
        Kotlin => &[("fun privacyFragment(){\n", "\n}")],
        C | Cpp => &[("void privacy_fragment(){\n", "\n}")],
    }
}

/// The released Kotlin grammar has gaps in generated Unicode identifier tables.
/// Alias valid Unicode identifier characters only in the analysis copy, retaining
/// their UTF-8 byte widths so AST ranges still address the original source. This
/// never changes delivered text and never aliases literals/comments or symbols
/// such as emoji into identifiers. No names or credential values are decoded.
fn kotlin_identifier_aliases(source: &str) -> String {
    let mut bytes = source.as_bytes().to_vec();
    let mut quote = None;
    let mut escaped = false;
    let mut line_comment = false;
    let mut block_depth = 0usize;
    let mut identifier = false;
    let mut chars = source.char_indices().peekable();
    while let Some((index, character)) = chars.next() {
        let next = chars.peek().map(|(_, value)| *value);
        if line_comment {
            line_comment = character != '\n';
        } else if block_depth > 0 {
            if character == '/' && next == Some('*') {
                block_depth += 1;
                chars.next();
            } else if character == '*' && next == Some('/') {
                block_depth -= 1;
                chars.next();
            }
        } else if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                quote = None;
            }
        } else if character == '/' && next == Some('/') {
            line_comment = true;
            identifier = false;
            chars.next();
        } else if character == '/' && next == Some('*') {
            block_depth = 1;
            identifier = false;
            chars.next();
        } else if matches!(character, '\'' | '"' | '`') {
            quote = Some(character);
            identifier = false;
        } else if character.is_ascii() {
            if character.is_ascii_alphabetic() || character == '_' {
                identifier = true;
            } else if !character.is_ascii_digit() {
                identifier = false;
            }
        } else if unicode_ident::is_xid_start(character)
            || (identifier && unicode_ident::is_xid_continue(character))
        {
            bytes[index] = b'u';
            bytes[index + 1..index + character.len_utf8()].fill(b'_');
            identifier = true;
        } else {
            identifier = false;
        }
    }
    String::from_utf8(bytes).expect("Unicode identifier aliasing preserves UTF-8 byte offsets")
}

fn is_environment_style_assignment(node: Node<'_>, source: &str) -> bool {
    // Nested member/call expressions must not supply instance-only evidence
    // for an otherwise opaque environment assignment either.
    let mut ancestor = node.parent();
    for _ in 0..32 {
        let Some(parent) = ancestor else {
            break;
        };
        if matches!(parent.kind(), "assignment" | "assignment_expression") {
            let Some(left) = parent.child_by_field_name("left") else {
                return false;
            };
            let name = &source[left.byte_range()];
            return name.bytes().any(|byte| byte.is_ascii_uppercase())
                && name.bytes().all(|byte| {
                    byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_'
                });
        }
        if !is_reference_expression(parent.kind()) && parent.kind() != "parenthesized_expression" {
            break;
        }
        ancestor = parent.parent();
    }
    false
}

fn is_literal_or_comment(kind: &str) -> bool {
    kind.contains("comment")
        || matches!(
            kind,
            "string"
                | "string_literal"
                | "raw_string_literal"
                | "interpreted_string_literal"
                | "template_string"
                | "concatenated_string"
                | "character_literal"
                | "char_literal"
                | "verbatim_string_literal"
                | "interpolated_string_expression"
                | "multiline_string_literal"
        )
}

fn is_reference_expression(kind: &str) -> bool {
    matches!(
        kind,
        "identifier"
            | "simple_identifier"
            | "type_identifier"
            | "predefined_type"
            | "primitive_type"
            | "generic_type"
            | "call"
            | "call_expression"
            | "method_invocation"
            | "invocation_expression"
            | "attribute"
            | "subscript"
            | "member_expression"
            | "index_expression"
            | "field_expression"
            | "selector_expression"
            | "field_access"
            | "member_access_expression"
            | "navigation_expression"
            | "macro_invocation"
            | "scoped_identifier"
            | "scoped_type_identifier"
            | "qualified_name"
            | "qualified_identifier"
    )
}

fn is_code_evidence(node: Node<'_>) -> bool {
    matches!(
        node.kind(),
        "function_definition"
            | "function_declaration"
            | "function_item"
            | "method_declaration"
            | "arrow_function"
            | "lambda_expression"
            | "call"
            | "call_expression"
            | "method_invocation"
            | "invocation_expression"
            | "let_declaration"
            | "variable_declarator"
            | "init_declarator"
            | "var_spec"
            | "short_var_declaration"
            | "property_declaration"
            | "local_variable_declaration"
            | "interface_declaration"
            | "return_statement"
            | "import_statement"
            | "import_declaration"
            | "macro_invocation"
    ) || (node.kind() == "assignment"
        && node
            .child_by_field_name("right")
            .is_some_and(|right| matches!(right.kind(), "dictionary" | "tuple" | "list")))
}

fn is_instance_member(node: Node<'_>, source: &str, language: SourceLanguage) -> bool {
    use SourceLanguage::*;
    let member_kind = match language {
        Python => "attribute",
        Javascript | Typescript | Tsx => "member_expression",
        Rust | Cpp => "field_expression",
        Java => "field_access",
        Csharp => "member_access_expression",
        Kotlin => "navigation_expression",
        // Go and C have no reserved instance receiver. Without code/path
        // context an arbitrary receiver name is not proof of a reference.
        Go | C => return false,
    };
    if node.kind() != member_kind || node.has_error() {
        return false;
    }
    let mut object = node;
    for _ in 0..32 {
        let receiver = match language {
            Python => {
                object.kind() == "identifier"
                    && matches!(&source[object.byte_range()], "self" | "cls")
            }
            Javascript | Typescript | Tsx | Java => matches!(object.kind(), "this" | "super"),
            Rust => object.kind() == "self",
            Csharp => matches!(object.kind(), "this" | "base"),
            Kotlin => matches!(object.kind(), "this_expression" | "super_expression"),
            Cpp => object.kind() == "this",
            Go | C => false,
        };
        if receiver {
            return true;
        }
        let next = if object.kind() == member_kind {
            match language {
                Python | Javascript | Typescript | Tsx | Java => {
                    object.child_by_field_name("object")
                }
                Rust => object.child_by_field_name("value"),
                Csharp => object.child_by_field_name("expression"),
                Kotlin => object.named_child(0),
                Cpp => object.child_by_field_name("argument"),
                Go | C => None,
            }
        } else if object.kind() == "parenthesized_expression" {
            object.named_child(0)
        } else {
            None
        };
        let Some(next) = next else {
            return false;
        };
        object = next;
    }
    false
}

fn source_regions(text: &str, hint: SourceHint) -> Vec<(Range<usize>, SourceHint, bool)> {
    // A known source file must be parsed as that file. Markdown fences or patch
    // markers inside its string literals must not become separate code regions.
    if matches!(hint, SourceHint::Language(_)) {
        return vec![(0..text.len(), hint, false)];
    }
    let mut fences = Vec::new();
    let mut fence: Option<(usize, SourceHint, char)> = None;
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim();
        let delimiter = if trimmed.starts_with("```") {
            Some('`')
        } else if trimmed.starts_with("~~~") {
            Some('~')
        } else {
            None
        };
        if let Some(delimiter) = delimiter {
            if let Some((start, language, open)) = fence {
                if open == delimiter {
                    fences.push((start..offset, language, false));
                    fence = None;
                }
            } else {
                let tag = trimmed
                    .trim_start_matches(delimiter)
                    .split_whitespace()
                    .next()
                    .unwrap_or("");
                let language = SourceLanguage::named(tag)
                    .map(SourceHint::Language)
                    .unwrap_or(SourceHint::Opaque);
                fence = Some((offset + line.len(), language, delimiter));
            }
        }
        offset += line.len();
    }
    if let Some((start, language, _)) = fence {
        fences.push((start..text.len(), language, false));
    }
    if !fences.is_empty() {
        return fences;
    }
    let mut patches = Vec::new();
    let mut active: Option<(usize, SourceHint)> = None;
    offset = 0;
    for line in text.split_inclusive('\n') {
        let target = [
            "*** Update File:",
            "*** Add File:",
            "*** Delete File:",
            "+++ b/",
        ]
        .iter()
        .find_map(|prefix| line.trim().strip_prefix(prefix));
        if let Some(target) = target {
            if let Some((start, language)) = active {
                patches.push((start..offset, language, true));
            }
            active = Some((offset + line.len(), SourceHint::from_path(target.trim())));
        }
        offset += line.len();
    }
    if let Some((start, language)) = active {
        patches.push((start..text.len(), language, true));
    }
    if !patches.is_empty() {
        patches
    } else {
        vec![(0..text.len(), hint, false)]
    }
}

fn normalize_patch(text: &str) -> String {
    let mut bytes = text.as_bytes().to_vec();
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if ["@@", "***", "diff ", "index ", "--- ", "+++ "]
            .iter()
            .any(|prefix| line.starts_with(prefix))
        {
            for byte in &mut bytes[offset..offset + line.len()] {
                if !matches!(*byte, b'\n' | b'\r') {
                    *byte = b' ';
                }
            }
        } else if line.starts_with(['+', '-']) {
            bytes[offset] = b' ';
        }
        offset += line.len();
    }
    String::from_utf8(bytes).expect("patch normalization preserves UTF-8")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const LANGUAGE_CASES: &[(&str, &str)] = &[
        ("auth.py", "def run(minted, callback=lambda value: value):\n    password = minted\n"),
        ("auth.js", "const run = ({credential: minted}) => { const password = minted; };"),
        ("auth.ts", "const {refresh_token: minted} = createToken(); const password = minted;"),
        ("auth.tsx", "const Run = ({minted}: {minted: string}) => { const password = minted; return <div/>; };"),
        ("auth.jsx", "function Run({minted}) { const password = minted; return <div/>; }"),
        ("auth.rs", "fn run((minted, _): (String, usize)) { let password = minted; }"),
        ("auth.go", "package auth\nfunc run() { minted := createToken()\nvar password string\npassword = minted\n_ = password\n}"),
        ("Auth.java", "class Auth { void run(String minted) { String password = minted; } }"),
        ("Auth.cs", "class Auth { void Run(string minted) { string password = minted; } }"),
        ("auth.kt", "fun run(minted: String) { val password = minted }"),
        ("auth.c", "void run(char *minted) { char *password = minted; }"),
        ("auth.cpp", "auto run(const std::string& minted) { auto password = minted; return password; }"),
        ("auth.h", "namespace auth { inline auto run(const std::string& minted) { auto password = minted; return password; } }"),
    ];

    // Partial edit payloads, not just complete files or bare bound identifiers.
    pub(crate) const MEMBER_CASES: &[(&str, &str, &str)] = &[
        (
            "auth.py",
            "        \"apiKey\": self.credentials.api_key,",
            "self.credentials.api_key",
        ),
        (
            "auth.js",
            "        \"apiKey\": this.credentials.apiKey,",
            "this.credentials.apiKey",
        ),
        (
            "auth.jsx",
            "        \"apiKey\": this.credentials.apiKey,",
            "this.credentials.apiKey",
        ),
        (
            "auth.ts",
            "        \"apiKey\": this.credentials.apiKey,",
            "this.credentials.apiKey",
        ),
        (
            "auth.tsx",
            "        \"apiKey\": this.credentials.apiKey,",
            "this.credentials.apiKey",
        ),
        (
            "auth.rs",
            "let api_key = self.credentials.api_key;",
            "self.credentials.api_key",
        ),
        (
            "auth.go",
            "var apiKey = client.credentials.apiKey",
            "client.credentials.apiKey",
        ),
        (
            "Auth.java",
            "String apiKey = this.credentials.apiKey;",
            "this.credentials.apiKey",
        ),
        (
            "Auth.cs",
            "string apiKey = this.Credentials.ApiKey;",
            "this.Credentials.ApiKey",
        ),
        (
            "auth.kt",
            "val apiKey = this.credentials.apiKey",
            "this.credentials.apiKey",
        ),
        (
            "auth.c",
            "const char *api_key = client->credentials.api_key;",
            "client->credentials.api_key",
        ),
        (
            "auth.cpp",
            "auto api_key = this->credentials.api_key;",
            "this->credentials.api_key",
        ),
        (
            "auth.h",
            "auto api_key = this->credentials.api_key;",
            "this->credentials.api_key",
        ),
    ];

    #[test]
    fn unbound_generated_looking_names_are_not_definite_in_known_files() {
        // `api_key = key_live_...` without quotes is a bare name to the
        // parser, but nothing binds it: it must stay a secret candidate in
        // known-language mode exactly as in Auto mode.
        let source = "api_key = key_live_aaaaaaaaaaaaaaaaaaaaaaaa\n";
        for hint in [SourceHint::from_path("auth.py"), SourceHint::Auto] {
            let references = SourceReferences::analyze(source, hint);
            let start = source.find("key_live").unwrap();
            assert!(
                !references.recognizes(
                    start,
                    start + "key_live_aaaaaaaaaaaaaaaaaaaaaaaa".len(),
                    true
                ),
                "{hint:?}: unbound name must not be a definite reference"
            );
        }
        // Imported and declared names are definite in both modes.
        for source in [
            "from vault import minted\napi_key = minted\n",
            "def run(minted):\n    api_key = minted\n",
        ] {
            let references = SourceReferences::analyze(source, SourceHint::from_path("auth.py"));
            let start = source.rfind("minted").unwrap();
            assert!(
                references.recognizes(start, start + "minted".len(), true),
                "{source}"
            );
        }
        // Scoped paths (Rust/C++/Java/C#) are references, not secrets.
        let rust = "fn run() { let api_key = crate::config::TOKEN; }";
        let references = SourceReferences::analyze(rust, SourceHint::from_path("auth.rs"));
        let start = rust.find("crate::config::TOKEN").unwrap();
        assert!(references.recognizes(start, start + "crate::config::TOKEN".len(), true));
    }

    #[test]
    fn all_registered_languages_preserve_partial_member_reference_offsets() {
        for (path, source, value) in MEMBER_CASES {
            for hint in [SourceHint::from_path(path), SourceHint::Auto] {
                let references = SourceReferences::analyze(source, hint);
                let start = source.rfind(value).unwrap();
                assert!(
                    references.recognizes(start, start + value.len(), true),
                    "{path}, {hint:?}: {source}"
                );
            }
        }
    }

    #[test]
    fn each_registered_grammar_recognizes_reference_positions() {
        for (path, source) in LANGUAGE_CASES {
            let references = SourceReferences::analyze(source, SourceHint::from_path(path));
            let start = source.rfind("minted").unwrap();
            assert!(
                references.contains(start, start + "minted".len()),
                "{path}: {source}"
            );
        }
    }

    #[test]
    fn unicode_identifiers_are_recognized_without_ascii_binding_regexes() {
        for (path, source) in [
            ("auth.py", "def run(临时令牌):\n    password = 临时令牌\n"),
            (
                "auth.ts",
                "const 临时令牌 = createToken(); const password = 临时令牌;",
            ),
            (
                "auth.kt",
                "fun run(临时令牌: String) { val password = 临时令牌 }",
            ),
        ] {
            let references = SourceReferences::analyze(source, SourceHint::from_path(path));
            let start = source.rfind("临时令牌").unwrap();
            assert!(
                references.contains(start, start + "临时令牌".len()),
                "{path}"
            );
        }
    }

    #[test]
    fn names_in_other_scopes_do_not_exempt_strings_or_comments() {
        let source = "def first(minted):\n    password = minted\n\ndef second():\n    password = \"minted\"\n    # password = minted\n";
        let references =
            SourceReferences::analyze(source, SourceHint::Language(SourceLanguage::Python));
        let actual = source.find("password = minted").unwrap() + "password = ".len();
        assert!(references.contains(actual, actual + 6));
        let literal = source.find("\"minted\"").unwrap() + 1;
        assert!(!references.contains(literal, literal + 6));
        let comment = source.rfind("minted").unwrap();
        assert!(!references.contains(comment, comment + 6));
    }

    #[test]
    fn unhinted_python_mapping_fragments_exempt_only_instance_members() {
        let source = "\"apiKey\": self.credentials.api_key, \"password\": literalpassword,";
        let references = SourceReferences::analyze(source, SourceHint::Auto);
        let start = source.find("self.credentials.api_key").unwrap();
        assert!(references.recognizes(start, start + "self.credentials.api_key".len(), true));
        let literal = source.find("literalpassword").unwrap();
        assert!(!references.contains(literal, literal + "literalpassword".len()));
        for text in [
            "\"apiKey\": \"self.api_key\",",
            "\"apiKey\": arbitrary.value,",
            "\"apiKey\": self.api_key + ,",
        ] {
            assert!(
                SourceReferences::analyze(text, SourceHint::Auto)
                    .expressions
                    .is_empty(),
                "{text}"
            );
        }
    }

    #[test]
    fn instance_member_fragments_work_in_each_grammar_without_synthetic_evidence() {
        for (path, source, member) in [
            (
                "auth.py",
                "\"apiKey\": cls.credentials.api_key,",
                "cls.credentials.api_key",
            ),
            (
                "auth.js",
                "\"apiKey\": super.credentials.apiKey,",
                "super.credentials.apiKey",
            ),
            (
                "auth.ts",
                "\"apiKey\": this?.credentials?.apiKey,",
                "this?.credentials?.apiKey",
            ),
            (
                "auth.tsx",
                "\"apiKey\": this.credentials.apiKey,",
                "this.credentials.apiKey",
            ),
            (
                "auth.rs",
                "api_key = self.credentials.api_key;",
                "self.credentials.api_key",
            ),
            (
                "Auth.java",
                "apiKey = super.credentials.apiKey;",
                "super.credentials.apiKey",
            ),
            (
                "Auth.cs",
                "apiKey = base.Credentials.ApiKey;",
                "base.Credentials.ApiKey",
            ),
            (
                "auth.kt",
                "apiKey = this.credentials.apiKey",
                "this.credentials.apiKey",
            ),
            (
                "auth.cpp",
                "api_key = (this)->credentials.api_key;",
                "(this)->credentials.api_key",
            ),
        ] {
            let SourceHint::Language(language) = SourceHint::from_path(path) else {
                panic!("missing grammar for {path}");
            };
            // Check the actual grammar rather than allowing auto routing to
            // recognize a shared spelling using a different language.
            let (expressions, definite, evidence) =
                parse_references(source, language, true, Instant::now(), &mut 0).unwrap();
            let start = source.find(member).unwrap();
            assert!(evidence, "{path}: {source}");
            assert!(
                expressions.contains(&(start..start + member.len())),
                "{path}: {source}"
            );
            assert!(
                definite.contains(&(start..start + member.len())),
                "{path}: {source}"
            );
        }
        for language in [
            SourceLanguage::Go,
            SourceLanguage::Java,
            SourceLanguage::Csharp,
            SourceLanguage::Kotlin,
            SourceLanguage::C,
            SourceLanguage::Cpp,
        ] {
            let source = "password = literalpassword;";
            if let Some((expressions, _, evidence)) =
                parse_references(source, language, true, Instant::now(), &mut 0)
            {
                assert!(!evidence, "synthetic wrapper evidence: {language:?}");
                assert!(
                    expressions.is_empty(),
                    "synthetic wrapper references: {language:?}"
                );
            }
        }
    }

    #[test]
    fn member_only_fragments_do_not_exempt_neighboring_unbound_words() {
        for source in [
            "\"apiKey\": this.credentials.apiKey, \"password\": literalpassword,",
            "{\"apiKey\": this.credentials.apiKey, \"password\": literalpassword}",
            "\"apiKey\": self.credentials.api_key, \"password\": literalpassword,",
        ] {
            let references = SourceReferences::analyze(source, SourceHint::Auto);
            let literal = source.find("literalpassword").unwrap();
            assert!(
                !references.contains(literal, literal + "literalpassword".len()),
                "{source}"
            );
        }
        for (source, value) in [
            (
                "PASSWORD=self.credentials.api_key",
                "self.credentials.api_key",
            ),
            (
                "PASSWORD=this.credentials.apiKey;",
                "this.credentials.apiKey",
            ),
            (
                "password=client.credentials.apiKey",
                "client.credentials.apiKey",
            ),
            (
                "password=client->credentials.api_key;",
                "client->credentials.api_key",
            ),
            ("\"apiKey\": arbitrary.value,", "arbitrary.value"),
            ("\"apiKey\": this.apiKey + ,", "this.apiKey"),
        ] {
            let references = SourceReferences::analyze(source, SourceHint::Auto);
            let start = source.find(value).unwrap();
            assert!(
                !references.recognizes(start, start + value.len(), false),
                "{source}"
            );
        }
    }

    #[test]
    fn data_unknown_languages_and_limits_have_no_source_exemptions() {
        let text = "minted = createToken()\npassword = minted";
        for path in [
            "auth.env",
            "auth.json",
            "auth.yaml",
            "auth.swift",
            "auth.rb",
        ] {
            assert!(
                SourceReferences::analyze(text, SourceHint::from_path(path))
                    .expressions
                    .is_empty(),
                "{path}"
            );
        }
        let large = format!("{}\n{text}", " ".repeat(MAX_SOURCE_BYTES));
        assert!(
            SourceReferences::analyze(&large, SourceHint::Language(SourceLanguage::Python))
                .expressions
                .is_empty()
        );
        for data in [
            "PASSWORD=minted",
            "{\"password\":minted}",
            "def invalid(:\n password = minted",
        ] {
            assert!(
                SourceReferences::analyze(data, SourceHint::Auto)
                    .expressions
                    .is_empty(),
                "{data}"
            );
        }
    }

    #[test]
    fn fences_are_isolated_and_patch_offsets_are_preserved() {
        let text = "```python\npassword = createToken()\n```\n```swift\npassword = createToken()\n```\nPASSWORD=minted";
        let references = SourceReferences::analyze(text, SourceHint::Auto);
        let supported = text.find("createToken()").unwrap();
        let unsupported = text.rfind("createToken()").unwrap();
        assert!(references.contains(supported, supported + 13));
        assert!(!references.contains(unsupported, unsupported + 13));
        let patch = "*** Update File: auth.py\n@@\n+body={\"refresh_token\": minted}\n*** Update File: auth.env\n+PASSWORD=minted";
        let references = SourceReferences::analyze(patch, SourceHint::Auto);
        let code = patch.find("minted").unwrap();
        let data = patch.rfind("minted").unwrap();
        assert!(references.contains(code, code + 6));
        assert!(!references.contains(data, data + 6));
    }

    #[test]
    fn markers_inside_known_source_literals_do_not_create_code_exemptions() {
        let source = "def run():\n    example = \"\"\"\n```python\npassword = minted\n```\n*** Update File: auth.py\n+password = minted\n\"\"\"\n";
        let references = SourceReferences::analyze(source, SourceHint::from_path("auth.py"));
        for (start, _) in source.match_indices("minted") {
            assert!(!references.contains(start, start + "minted".len()));
        }
    }

    #[test]
    fn hinted_partial_fragments_use_language_wrappers_without_offset_shifts() {
        for (path, fragment) in [
            (
                "auth.go",
                "minted := createToken()\nvar password string\npassword = minted",
            ),
            (
                "Auth.java",
                "String minted = createToken();\nString password = minted;",
            ),
            (
                "Auth.cs",
                "string minted = CreateToken();\nstring password = minted;",
            ),
            ("auth.py", "\"refresh_token\": minted,"),
            ("auth.ts", "refresh_token: minted,"),
            (
                "auth.rs",
                "let minted = create_token();\nlet password = minted;",
            ),
        ] {
            let patch = format!(
                "*** Update File: {path}\n@@\n{}",
                fragment
                    .lines()
                    .map(|line| format!("+{line}\n"))
                    .collect::<String>()
            );
            let references = SourceReferences::analyze(&patch, SourceHint::Auto);
            let start = patch.rfind("minted").unwrap();
            assert!(references.contains(start, start + 6), "{path}: {patch}");
        }
    }

    #[test]
    fn kotlin_unicode_analysis_aliases_never_modify_literals_comments_or_delivered_source() {
        let source = "fun run(临时令牌: String) { val password = 临时令牌; val note = \"临时令牌\"; /* 临时令牌 */ }";
        let aliased = kotlin_identifier_aliases(source);
        assert_eq!(aliased.len(), source.len());
        assert!(aliased.contains("\"临时令牌\""));
        assert!(aliased.contains("/* 临时令牌 */"));
        assert_eq!(
            kotlin_identifier_aliases("val password = 🔐🙂"),
            "val password = 🔐🙂"
        );
        let references =
            SourceReferences::analyze(source, SourceHint::Language(SourceLanguage::Kotlin));
        let literal = source.find("\"临时令牌\"").unwrap() + 1;
        assert!(!references.contains(literal, literal + "临时令牌".len()));
    }

    #[test]
    fn bound_long_identifiers_are_confident_only_in_their_lexical_scope() {
        let name = "generatedRefreshTokenForCurrentUser";
        let source = format!(
            "def first({name}):\n    password = {name}\n\ndef second():\n    password = {name}\n"
        );
        let references = SourceReferences::analyze(&source, SourceHint::Auto);
        let first = source.find(&format!("password = {name}")).unwrap() + "password = ".len();
        let second = source.rfind(name).unwrap();
        assert!(references.recognizes(first, first + name.len(), true));
        assert!(!references.recognizes(second, second + name.len(), true));
    }

    #[test]
    fn node_or_time_exhaustion_grants_no_speculative_reference_exemptions() {
        let source = format!("{}password=minted\n", "call();\n".repeat(20_000));
        assert!(source.len() < MAX_SOURCE_BYTES);
        let references =
            SourceReferences::analyze(&source, SourceHint::Language(SourceLanguage::Python));
        assert!(references.expressions.is_empty());
        assert!(references.definite.is_empty());
    }
}
