import cors from "cors";
import express from "express";
import multer from "multer";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pythonApi = process.env.PYTHON_API_URL || "http://127.0.0.1:8000";
const allowedOrigin = process.env.CLIENT_ORIGIN || true;
const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 25 * 1024 * 1024 } });

app.use(cors({ origin: allowedOrigin }));
app.use(express.json({ limit: "1mb" }));

app.get("/api/health", async (_req, res) => {
  try {
    const response = await fetch(`${pythonApi}/health`);
    res.status(response.status).json(await response.json());
  } catch {
    res.status(503).json({ status: "unavailable" });
  }
});

app.post("/api/upload/:category", upload.single("file"), async (req, res) => {
  if (!req.file) return res.status(400).json({ detail: "A file is required" });
  const form = new FormData();
  form.append("file", new Blob([req.file.buffer]), req.file.originalname);
  try {
    const response = await fetch(`${pythonApi}/upload/${req.params.category}`, {
      method: "POST",
      body: form,
    });
    res.status(response.status).json(await response.json());
  } catch (error) {
    res.status(502).json({ detail: `Python API unavailable: ${error.message}` });
  }
});

for (const route of ["/rebuild", "/chat", "/interview/question", "/interview/evaluate"]) {
  app.post(`/api${route}`, async (req, res) => {
    try {
      const response = await fetch(`${pythonApi}${route}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req.body),
      });
      res.status(response.status).json(await response.json());
    } catch (error) {
      res.status(502).json({ detail: `Python API unavailable: ${error.message}` });
    }
  });
}

const port = Number(process.env.PORT || 3001);
app.listen(port, () => console.log(`Express gateway listening on http://127.0.0.1:${port}`));
