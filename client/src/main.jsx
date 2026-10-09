import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API = (import.meta.env.VITE_API_URL || "/api").replace(/\/$/, "");

async function request(url, options = {}) {
  const response = await fetch(`${API}${url}`, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || "Request failed");
  return payload;
}

function Sources({ sources = [] }) {
  return <details><summary>Sources ({sources.length})</summary>
    {sources.map((source, index) => <p key={`${source.source}-${index}`}><b>{index + 1}. {source.source}</b> · chunk {source.chunk}</p>)}
  </details>;
}

function App() {
  const [mode, setMode] = useState("chat");
  const [message, setMessage] = useState("");
  const [result, setResult] = useState(null);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [focus, setFocus] = useState("candidate projects and technical decisions");
  const [uploads, setUploads] = useState({ resume: null, projects: null, job_descriptions: null });
  const [status, setStatus] = useState("");

  async function upload(category) {
    const file = uploads[category];
    if (!file) return setStatus("Choose a file first.");
    const form = new FormData();
    form.append("file", file);
    setStatus(`Uploading ${file.name}...`);
    await request(`/upload/${category}`, { method: "POST", body: form });
    setStatus(`${file.name} uploaded. Rebuilding index...`);
    const rebuilt = await request("/rebuild", { method: "POST" });
    setStatus(`Indexed ${rebuilt.vectors} vectors.`);
  }

  async function ask() {
    setStatus("Thinking...");
    setResult(await request("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: message, k: 3 }) }));
    setStatus("");
  }

  async function generate() {
    setStatus("Generating question...");
    const generated = await request("/interview/question", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ focus, k: 5 }) });
    setQuestion(generated.question);
    setResult(generated);
    setStatus("");
  }

  async function evaluate() {
    setStatus("Evaluating answer...");
    setResult(await request("/interview/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, answer, source_paths: result.sources.map((source) => source.source) }) }));
    setStatus("");
  }

  return <main>
    <h1>AI Interview & Resume RAG Assistant</h1>
    <p>React + Express + Python LangChain backend</p>
    <section className="uploads">
      {Object.keys(uploads).map((category) => <div className="upload" key={category}>
        <label>{category.replace("_", " ")}</label>
        <input type="file" onChange={(event) => setUploads({ ...uploads, [category]: event.target.files[0] })} />
        <button onClick={() => upload(category)}>Upload & index</button>
      </div>)}
    </section>
    <nav><button className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>Chat</button><button className={mode === "interview" ? "active" : ""} onClick={() => setMode("interview")}>Interview</button></nav>
    {mode === "chat" ? <section>
      <h2>Ask about your documents</h2>
      <textarea value={message} onChange={(event) => setMessage(event.target.value)} placeholder="What Python experience is required?" />
      <button onClick={ask}>Ask</button>
      {result?.answer && <article><h3>Answer</h3><p>{result.answer}</p><Sources sources={result.sources} /></article>}
    </section> : <section>
      <h2>Interview practice</h2>
      <input value={focus} onChange={(event) => setFocus(event.target.value)} />
      <button onClick={generate}>Generate question</button>
      {question && <><h3>{question}</h3><textarea value={answer} onChange={(event) => setAnswer(event.target.value)} placeholder="Write your answer..." /><button onClick={evaluate}>Evaluate answer</button></>}
      {result?.score !== undefined && <article><h3>Score: {result.score}/10</h3><p>{result.feedback}</p><b>Correct points</b>{result.correct_points.map((point) => <p key={point}>✓ {point}</p>)}<b>Missing points</b>{result.missing_points.map((point) => <p key={point}>- {point}</p>)}</article>}
    </section>}
    {status && <footer>{status}</footer>}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
