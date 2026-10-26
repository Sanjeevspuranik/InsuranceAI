"use client";

import { FormEvent, useEffect, useState } from "react";
import { signIn, signOut, useSession } from "next-auth/react";
import ReactMarkdown from "react-markdown";
import CostEstimatePanel, {
  ClaimBreakdown,
  CostEstimate,
  InternalEstimate,
} from "./CostEstimatePanel";

type Claim = {
  claim_id: string;
  status: "queued" | "processing" | "completed" | "failed";
  stage?: string | null;
  progress_pct?: number | null;
  report?: string | null;
  rationale?: string | null;
  reasoning_tokens?: number | null;
  cost_estimate?: CostEstimate | null;
  internal_damage_estimate?: InternalEstimate | null;
  claim_breakdown?: ClaimBreakdown | null;
  error?: string | null;
  report_pdf_url?: string | null;
  annotated_image_url?: string | null;
};

const apiBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const authMode = process.env.NEXT_PUBLIC_AUTH_MODE ?? "development";

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string };
    return body.detail ?? `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

export default function Home() {
  const { data: session, status: sessionStatus } = useSession();
  const [file, setFile] = useState<File | null>(null);
  const [claim, setClaim] = useState<Claim | null>(null);
  const [annotatedUrl, setAnnotatedUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState("");
  const accessToken = session?.apiAccessToken;
  const isAuthenticated = Boolean(accessToken);
  // Short step name for the button; the full detail stays in the status line.
  const stepLabel = `${message.split(":")[0].replace(/\.\.\.$/, "") || "Analyzing"}${
    progress === null ? "" : ` · ${progress}%`
  }`;

  useEffect(
    () => () => {
      if (annotatedUrl) URL.revokeObjectURL(annotatedUrl);
    },
    [annotatedUrl],
  );

  async function fetchClaim(claimId: string, token: string): Promise<Claim> {
    const response = await fetch(`${apiBaseUrl}/api/v1/claims/${claimId}`, {
      headers: { Authorization: `Bearer ${token}` },
      cache: "no-store",
    });
    if (!response.ok) throw new Error(await readError(response));
    return (await response.json()) as Claim;
  }

  async function loadAnnotatedImage(url: string, token: string) {
    const response = await fetch(`${apiBaseUrl}${url}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) throw new Error(await readError(response));
    const objectUrl = URL.createObjectURL(await response.blob());
    setAnnotatedUrl(objectUrl);
  }

  async function submitClaim(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file || !accessToken) return;

    setBusy(true);
    setError("");
    setMessage("Uploading image...");
    setProgress(0);
    setClaim(null);
    setAnnotatedUrl(null);

    try {
      const form = new FormData();
      form.append("file", file);
      const upload = await fetch(`${apiBaseUrl}/api/v1/claims`, {
        method: "POST",
        headers: { Authorization: `Bearer ${accessToken}` },
        body: form,
      });
      if (!upload.ok) throw new Error(await readError(upload));
      const accepted = (await upload.json()) as { claim_id: string };
      setMessage("Claim queued. Waiting for an analysis worker...");
      setProgress(2);

      let latest: Claim | undefined;
      for (let attempt = 0; attempt < 300; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 1500));
        latest = await fetchClaim(accepted.claim_id, accessToken);
        setClaim(latest);
        if (latest.status === "completed") break;
        if (latest.status === "failed") {
          throw new Error(latest.error ?? "Claim analysis failed.");
        }
        if (latest.status === "queued") {
          setMessage("Waiting for an available analysis worker...");
          setProgress(2);
        } else {
          setMessage(latest.stage ?? "Analyzing image and preparing the report...");
          setProgress(latest.progress_pct ?? null);
        }
      }
      if (!latest || latest.status !== "completed") {
        throw new Error("Analysis is taking longer than expected. Check this claim again shortly.");
      }
      setProgress(100);
      setMessage("Analysis complete.");
      if (latest.annotated_image_url) {
        await loadAnnotatedImage(latest.annotated_image_url, accessToken);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unexpected request error.");
      setMessage("");
    } finally {
      setBusy(false);
    }
  }

  async function downloadPdf() {
    if (!claim?.report_pdf_url || !accessToken) return;
    setError("");
    try {
      const response = await fetch(`${apiBaseUrl}${claim.report_pdf_url}`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!response.ok) throw new Error(await readError(response));
      const objectUrl = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `claim-${claim.claim_id}.pdf`;
      anchor.click();
      URL.revokeObjectURL(objectUrl);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not download the PDF.");
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <h1>InsuranceAI Claims</h1>
          <span className="muted">Vehicle damage assessment workspace</span>
        </div>
        {isAuthenticated ? (
          <button className="button secondary" onClick={() => signOut()}>
            Sign out
          </button>
        ) : null}
      </header>

      {!isAuthenticated ? (
        <section className="card">
          <h2>Sign in to continue</h2>
          <p className="muted">
            {sessionStatus === "loading"
              ? "Checking your session..."
              : "Authentication is required to upload or view claim data."}
          </p>
          {sessionStatus !== "loading" ? (
            <button
              className="button"
              onClick={() =>
                signIn(authMode === "development" ? "local-dev" : "keycloak")
              }
            >
              {authMode === "development" ? "Sign in (local development)" : "Sign in"}
            </button>
          ) : null}
        </section>
      ) : (
        <>
          <section className="card">
            <h2>Start a damage assessment</h2>
            <p className="muted">
              Upload one JPEG, PNG, or WEBP image. The report includes
              indicative repair-cost ranges, which are not a settlement.
            </p>
            <form onSubmit={submitClaim} aria-busy={busy}>
              <input
                className="upload"
                type="file"
                accept="image/jpeg,image/png,image/webp"
                onChange={(event) => setFile(event.currentTarget.files?.[0] ?? null)}
                required
              />
              <button className="button" disabled={busy || !file}>
                {busy ? (
                  <span className="button-label">
                    <span className="spinner" aria-hidden="true" />
                    {stepLabel}
                  </span>
                ) : (
                  "Upload and analyze"
                )}
              </button>
              {busy ? (
                <div
                  className="progress-track"
                  role="progressbar"
                  aria-label="Claim analysis progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={progress ?? undefined}
                  aria-valuetext={message || "Claim analysis in progress"}
                >
                  <span
                    className={`progress-indicator${progress === null ? "" : " determinate"}`}
                    style={progress === null ? undefined : { width: `${Math.max(progress, 3)}%` }}
                  />
                </div>
              ) : null}
            </form>
            {message ? <p className="status" role="status">{message}</p> : null}
            {error ? <p className="error" role="alert">{error}</p> : null}
          </section>

          {claim ? (
            <section className="card">
              <div className="row">
                <div>
                  <h2>Claim {claim.claim_id.slice(0, 8)}</h2>
                  <p className="status">Status: {claim.status}</p>
                </div>
                {claim.status === "completed" ? (
                  <button className="button" onClick={downloadPdf}>
                    Download PDF
                  </button>
                ) : null}
              </div>
              {claim.rationale ? (
                <div>
                  <h3>Assessment rationale</h3>
                  <p>{claim.rationale}</p>
                </div>
              ) : null}
              {claim.reasoning_tokens ? (
                <p className="muted">
                  Provider-reported reasoning tokens: {claim.reasoning_tokens}
                </p>
              ) : null}
              {annotatedUrl ? (
                <div>
                  <h3>Annotated damage image</h3>
                  {/* The image URL is an authorized, short-lived browser object URL. */}
                  <img className="annotated" src={annotatedUrl} alt="AI-annotated vehicle damage" />
                </div>
              ) : null}
              {claim.status === "completed" && accessToken ? (
                <CostEstimatePanel
                  claimId={claim.claim_id}
                  apiBaseUrl={apiBaseUrl}
                  token={accessToken}
                  costEstimate={claim.cost_estimate}
                  internalEstimate={claim.internal_damage_estimate}
                  breakdown={claim.claim_breakdown}
                  onBreakdownChange={(breakdown) =>
                    setClaim((current) =>
                      current ? { ...current, claim_breakdown: breakdown } : current,
                    )
                  }
                />
              ) : null}
              {claim.report ? (
                <article className="report">
                  <ReactMarkdown>{claim.report}</ReactMarkdown>
                </article>
              ) : null}
            </section>
          ) : null}
        </>
      )}
    </main>
  );
}
