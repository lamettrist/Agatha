# Video upload integration

Both `POST /api/scenes` and `POST /api/scenes/video` accept video uploads under
multipart `file`, `image`, or `video`. The generic endpoint routes by the filename
extension as well as the media type, including `application/octet-stream` uploads.

```javascript
const data = new FormData();
data.append('file', videoFile);
const response = await fetch(API_BASE + '/api/scenes/video', {
  method: 'POST', body: data
});
const result = await response.json();
if (!response.ok) throw new Error(JSON.stringify(result.detail ?? result));
if (!result.scene || !Array.isArray(result.scene.objects)) {
  throw new Error('The API returned an unexpected reconstruction response.');
}
const bodies = result.scene.objects;
if (bodies.length === 0) {
  throw new Error(result.report?.warnings?.join(' ') ||
    'No bodies were reconstructed. Inspect the reconstruction report.');
}
// Use bodies for object counts, confidence and extents.
// Body positions and dimensions use meters and right-handed Y-up coordinates.
```

Do not treat HTTP errors, unavailable endpoints, or absent `scene.objects` as a
successful reconstruction with zero objects. The API server must run the current
checkout in the CUDA-compatible `.venv-reconstruction` environment. Restart a
previously running server after upgrading its source.

The local viewer is `/` on the same server. Its wording differs from “Detected
world model”; an external viewer must map the response above explicitly. It must
also allow long-running video requests (up to three minutes).

`GET /api/health` reports the supported media, endpoint paths, object-array field,
loaded models and CUDA availability. `https://precognition.base44.app` is allowed
by the default CORS configuration; an explicit `CORS_ORIGINS` setting still takes
precedence.

An empty video reconstruction returns HTTP 422, or 503 when detection is unavailable,
with its reconstruction report in the error detail. A successful response contains
the actual scene and may still have `report.quality_status = "needs_review"`.
That status describes failed measured quality checks; object count is not an
accuracy measurement.

Completed video runs with valid propagated tracks are cached by input content and
pipeline configuration. Empty or degraded tracking runs are recomputed on retry.
The original run duration remains in `report.duration_s`; a cached request reports
its own time separately in `report.request_duration_s`.

Measured fixture results and actual HTTP/cache receipts are saved in
`docs/video-reconstruction-report.json` and `docs/video-http-validation.json`.
