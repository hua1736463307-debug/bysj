# RuoYi-style UI release — 2026-11-02

Baseline: GitHub graduation/cxr-assist-20261027 at c966f7c, fetched before implementation.
Includes existing local P0/P2/P4 commits 02123fc and 1f48732 by cherry-pick; their original worktree remains unchanged.

Scope: frontend/src/App.vue, frontend/src/styles/ruoyi-theme.css, frontend/src/views/Login.vue and this receipt.
Behavior: dark collapsible navigation, breadcrumbs, visited-page tabs, account logout, responsive shell and medical platform branding.
Protected: all six CXR routes and six BCCD routes, login API, authentication guards and existing domain components remain in place.
Excluded: database migrations, data, model files, credentials, runtime containers and original uncommitted deploy/compose.yml.

Validation: production Vite build passed; all 13 existing frontend tests passed. Build reports the existing large-bundle warning.
Browser visual and authenticated end-to-end checks have not been run for this release.
Delivery: dedicated graduation/ruoyi-ui-20261102 branch; no main-branch overwrite or PR merge.
Rollback: return to baseline c966f7c; original local source remains at 1f48732.
