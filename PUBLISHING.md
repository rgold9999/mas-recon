# Distribution options

The package builds cleanly (`python -m build`) and the resulting wheel installs and runs correctly from a totally fresh virtualenv — verified, not assumed. What's left is picking where it actually lives, which is a decision about who can see and install it, not an engineering one. Three realistic options:

## Option 1: GitHub, pip-installable by URL

Lowest friction, no account/registry setup, fully reversible (delete the repo).

```bash
pip install "git+https://github.com/<org>/<repo>.git#egg=mas-recon[langchain]"
```

Good fit if: this stays inside Cisco/Outshift's own GitHub org (private repo) for now, or you're comfortable with it being visible if the org is public. Needs: a repo to push to — your call on org/visibility.

## Option 2: An internal/private package index

If Cisco has an internal Artifactory, Nexus, or similar PyPI-compatible registry, this publishes the same way a public PyPI release would, just pointed at an internal index URL. Keeps it fully inside the org, no external visibility at all, and gets `pip install mas-recon` to just work for anyone with network access to that index.

Needs: the internal index URL and credentials — not something I have or should guess at.

## Option 3: Public PyPI

`pip install mas-recon` for literally anyone, no auth needed. This is also the only option that's a genuinely public, hard-to-fully-reverse action (PyPI doesn't allow re-uploading a deleted version number, and the name becomes claimed) — worth being deliberate about, especially since the Thrad.ai/TradingAgents/CAIPE example scripts in this repo reference a specific internal gateway URL (`litellm.prod.outshift.ai`) that would ship as example code if included as-is.

Needs: a PyPI account for whichever org/identity should own the package name, and a decision on whether the example scripts get scrubbed of the internal gateway URL first (recommended regardless of where this goes, since those examples are meant to show the pattern, not point at an endpoint only Outshift can reach).

## What I'd do before any of these

Regardless of which option, two things are worth doing first:
1. **Scrub the internal gateway URL from the example scripts** (or make it clearly a placeholder) — right now they default to `https://litellm.prod.outshift.ai`, which is fine for internal use but shouldn't ship as the default in anything externally visible.
2. **Bump past 0.1.0** only once there's a real answer to "where does this live" — version numbers on a public index are permanent.

## What I need from you

Which option, and for Option 1 or 2, the actual destination (repo name/org, or internal index URL) — I can push the actual release once that's decided, but publishing anything public is exactly the kind of action worth confirming explicitly rather than guessing.
