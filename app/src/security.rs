//! Navigation is independent of CSP: remote pages must never replace the UI.
pub fn navigation_allowed(url: &tauri::Url, development: bool) -> bool {
    if !url.username().is_empty() || url.password().is_some() {
        return false;
    }
    match (url.scheme(), url.host_str(), url.port()) {
        ("tauri", Some("localhost"), None) => true,
        ("http" | "https", Some("tauri.localhost"), None) => true,
        ("http", Some("localhost" | "127.0.0.1"), Some(1420)) => development,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_registered_command_is_in_the_permission_manifest() {
        let source = include_str!("lib.rs");
        let registered = source
            .split("tauri::generate_handler![")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        let build = include_str!("../build.rs");
        let manifest = build
            .split(".commands(&[")
            .nth(1)
            .unwrap()
            .split(']')
            .next()
            .unwrap();
        let parse = |text: &str| {
            text.split(',')
                .map(|s| s.trim().trim_matches('"').to_owned())
                .filter(|s| !s.is_empty())
                .collect::<std::collections::BTreeSet<_>>()
        };
        assert_eq!(
            parse(registered),
            parse(manifest),
            "new commands must be capability-controlled"
        );
    }
    #[test]
    fn release_policy_has_no_remote_or_dynamic_script_access() {
        let config: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let csp = config["app"]["security"]["csp"].as_str().unwrap();
        let directives: std::collections::HashMap<_, _> = csp
            .split(';')
            .map(str::trim)
            .filter_map(|s| s.split_once(' '))
            .collect();
        assert_eq!(directives["script-src"], "'self'");
        assert_eq!(directives["default-src"], "'none'");
        for directive in [
            "object-src",
            "base-uri",
            "frame-src",
            "frame-ancestors",
            "form-action",
        ] {
            assert_eq!(directives[directive], "'none'");
        }
        assert!(!directives["connect-src"].contains('*'));
        assert!(!directives["connect-src"].contains("ws:"));
        assert!(!directives["connect-src"].contains("https:"));
        assert!(directives["connect-src"].contains("http://127.0.0.1:8989"));
        let capability: serde_json::Value =
            serde_json::from_str(include_str!("../capabilities/default.json")).unwrap();
        assert!(capability.get("remote").is_none());
        assert_eq!(capability["windows"], serde_json::json!(["main"]));
        for permission in capability["permissions"].as_array().unwrap() {
            let permission = permission.as_str().unwrap();
            // App commands are "allow-<command>". A plugin contributes its own
            // permission set instead, named "<plugin>:<set>"; each is listed
            // here by name rather than allowed by shape, so adding one is a
            // decision somebody makes on purpose.
            assert!(
                permission.starts_with("allow-") || permission == "notification:default",
                "unreviewed permission {permission}"
            );
        }
    }
    #[test]
    fn only_app_origins_can_navigate() {
        for url in [
            "tauri://localhost/index.html#/sessions",
            "http://tauri.localhost/",
            "https://tauri.localhost/",
        ] {
            assert!(navigation_allowed(&url.parse().unwrap(), false));
        }
        for url in [
            "https://example.com",
            "http://127.0.0.1:8989",
            "http://tauri.localhost.evil/",
            "http://user@tauri.localhost/",
            "file:///etc/passwd",
            "data:text/html,test",
            "javascript:alert(1)",
            "http://tauri.localhost:9000/",
        ] {
            assert!(!navigation_allowed(&url.parse().unwrap(), true), "{url}");
        }
        let dev = "http://localhost:1420/".parse().unwrap();
        assert!(navigation_allowed(&dev, true));
        assert!(!navigation_allowed(&dev, false));
    }
}
