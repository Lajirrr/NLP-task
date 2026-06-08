# Translation Demo Frontend Design

Date: 2026-06-08

## Goal

Build a local web demo for video presentation of the Transformer English-to-Chinese translator. The demo focuses on one English sentence at a time: the user types English in the top panel, clicks translate, and sees the model output in the lower panel.

The visual direction is the selected "clean translation card" style: close to a Google Translate card, with a calm off-white page, large readable text, input above output, and minimal surrounding controls.

## Chosen Model

The demo defaults to the strongest restored character-level ensemble:

- `checkpoints/char-enhanced/averaged.pt`
- `checkpoints/char-adam98-e80/best.pt`
- `checkpoints/char-tied256-e60/averaged.pt`

Default decoding parameters:

- device: `cuda` when available, otherwise configurable fallback
- beam size: `4`
- length penalty: `1.5`
- no-repeat ngram size: `2`
- max decode length: `64`
- suppress `<UNK>` by default

This matches the previously recorded full-test setup with about `0.36` BLEU and keeps the demo output quality as high as the current from-scratch model allows.

## Architecture

Use a lightweight local Python server with no new runtime dependency. Python's standard `http.server` module is sufficient for a local-only demo, and the official documentation warns it is not intended for production use. That is acceptable because this is a controlled presentation tool, not a deployed service.

New files:

- `code/demo_server.py`
- `demo_frontend/index.html`
- `demo_frontend/styles.css`
- `demo_frontend/app.js`

`code/demo_server.py` will:

- load the ensemble once at process startup
- serve static files from `demo_frontend/`
- expose `POST /api/translate`
- return UTF-8 JSON responses
- print the local URL after startup

The frontend will stay dependency-free: plain HTML, CSS, and JavaScript.

## Data Flow

1. User opens the local URL, for example `http://localhost:8000`.
2. Browser loads `demo_frontend/index.html`.
3. User enters a sentence in the top textarea.
4. Frontend sends:

```json
{"text":"tom is a student ."}
```

to `POST /api/translate`.

5. Server tokenizes the source sentence with the existing `translate_text` path and decodes through the ensemble model.
6. Server responds:

```json
{
  "translation": "translated Chinese text",
  "source_tokens": ["tom", "is", "a", "student", "."],
  "prediction_tokens": ["..."]
}
```

7. Frontend renders the Chinese translation in the lower result panel.

## UI Design

The first screen is the actual translator, not a landing page.

Layout:

- centered vertical translation surface
- compact language row at the top: `English` and `Chinese (Simplified)`
- top input card with large textarea
- small centered swap/arrow visual between panels, decorative only
- lower result card with large translated text
- translate button near the input controls

Video-friendly behavior:

- large type for both input and output
- no dense debug output on screen
- subtle loading state while decoding
- result fades or slides in after completion
- layout remains stable before and after translation

Expected visible states:

- empty input: placeholder text, disabled or gently inert translate action
- loading: button label and result panel show translation in progress
- success: result text appears in the lower panel
- error: concise message in the result panel

## Error Handling

Backend:

- return `400` for blank input
- return `404` for unknown routes
- return `405` for unsupported methods
- return `500` with a short JSON error for unexpected model/server failures
- keep console logs useful but not noisy

Frontend:

- disable duplicate submissions while a request is pending
- show a readable error message if the request fails
- preserve the user's input after failure
- avoid rendering raw stack traces

## Testing

Unit-level tests:

- add a focused test for the translation API handler where practical, using a stub translator to avoid loading checkpoints
- verify blank input returns a client error
- verify successful translation returns UTF-8 JSON with the expected fields

Manual verification:

- start the server from the conda `base` environment
- confirm the URL loads in the browser
- translate `tom is a student .`
- confirm Chinese characters render correctly
- confirm a longer sentence does not break the layout
- confirm the console does not show request failures

## Out of Scope

- user authentication
- batch translation
- dataset browsing
- training controls
- production deployment
- replacing the current model

## Launch Command

From the project root:

```powershell
& "C:\Users\86133\anaconda3\shell\condabin\conda-hook.ps1"
conda activate base
python code\demo_server.py --device cuda
```

The server should print the final local URL for the demo.
