# static

Frontend static assets (UI pages, styling, client-side resources).

Primary entry file:
- `index.html`

Current user modes in UI:
- Claim
- Image
- PDF

## Runtime API endpoint

Frontend calls API routes under `/api/*` by default (same-origin).

For split deployment (frontend on Vercel/GitHub Pages, backend on Azure), set:
- `static/config.js`
- `window.__FACTLENS_API_BASE__ = "https://<your-azure-app>.azurewebsites.net"`

You can also override temporarily via query param:
- `?api_base=https://<your-azure-app>.azurewebsites.net`

## GitHub Pages

This repo includes a GitHub Actions workflow that deploys `static/` to Pages on pushes to `main`.

Before using the Pages site, set a secure API base in `static/config.js`, for example:
- `https://fact-lens-indic.vercel.app` (recommended proxy route), or
- `https://<your-api-domain>` (direct backend over HTTPS)

Do not use `http://...` API endpoints from a Pages site because browsers block HTTPS -> HTTP mixed content.

## Scope Snapshot
- Path: `static/README.md`.
- Purpose: Frontend static asset layout and UI entrypoints.
- Audience: Engineers running, extending, or evaluating this module.

