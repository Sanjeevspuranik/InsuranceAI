"use client";

import { FormEvent, useEffect, useState } from "react";

export type Range = { low: number; high: number } | null;

export type PriceSource = { url: string; title: string; retrieved_at: string };

export type EstimateItem = {
  part: string;
  operation: string;
  subtotal: Range;
  estimable: boolean;
  confidence: "none" | "low" | "medium";
  likelihood?: "low" | "medium" | "high";
  reason?: string;
  missing_components?: string[];
  sources: PriceSource[];
};

export type CostEstimate = {
  status: "estimated" | "unavailable";
  reason?: string;
  vehicle?: string;
  items?: EstimateItem[];
  total?: Range;
  assumptions?: string[];
  provider?: string | null;
  generated_at?: string;
  disclaimer?: string;
};

export type InternalEstimate = {
  status: string;
  items?: EstimateItem[];
  total?: Range;
  note?: string;
};

export type ClaimBreakdown = {
  template: string;
  covered: boolean;
  gross_repair: Range;
  depreciation: Range;
  deductible_inr: number;
  copay_pct: number;
  claimable: Range;
  possible_total_loss: boolean;
  assumptions: string[];
  disclaimer: string;
};

type PolicyTemplate = { id: string; label: string; description: string };

type Props = {
  claimId: string;
  apiBaseUrl: string;
  token: string;
  costEstimate?: CostEstimate | null;
  internalEstimate?: InternalEstimate | null;
  breakdown?: ClaimBreakdown | null;
  onBreakdownChange: (breakdown: ClaimBreakdown) => void;
};

const DISCLAIMER =
  "Indicative estimate only, not a settlement or approval. Final amounts depend on physical inspection, your actual policy terms and insurer review.";

const inr = (value: number) => `₹${Math.round(value).toLocaleString("en-IN")}`;
const formatRange = (range: Range | undefined) =>
  range ? `${inr(range.low)} – ${inr(range.high)}` : "Not estimable";

function ItemList({ items }: { items: EstimateItem[] }) {
  return (
    <ul className="estimate-list">
      {items.map((item, index) => (
        <li key={`${item.part}-${index}`}>
          <div className="row">
            <strong>
              {item.part} <span className="muted">({item.operation})</span>
            </strong>
            <span>{formatRange(item.subtotal)}</span>
          </div>
          <div className="estimate-meta">
            {item.likelihood ? (
              <span className={`badge likelihood-${item.likelihood}`}>
                {item.likelihood} likelihood
              </span>
            ) : null}
            {item.estimable ? (
              <span className={`badge confidence-${item.confidence}`}>
                {item.confidence} confidence
              </span>
            ) : (
              <span className="badge confidence-none">price not found</span>
            )}
            {item.reason ? <span className="muted">{item.reason}</span> : null}
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function CostEstimatePanel({
  claimId,
  apiBaseUrl,
  token,
  costEstimate,
  internalEstimate,
  breakdown,
  onBreakdownChange,
}: Props) {
  const [templates, setTemplates] = useState<PolicyTemplate[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    template: "standard_comprehensive",
    idv_inr: "",
    vehicle_age_months: "0",
    engine_cc: "1200",
    voluntary_deductible_inr: "0",
    copay_pct: "",
  });

  useEffect(() => {
    let cancelled = false;
    fetch(`${apiBaseUrl}/api/v1/policy-templates`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((response) => (response.ok ? response.json() : []))
      .then((data: PolicyTemplate[]) => {
        if (!cancelled) setTemplates(data);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [apiBaseUrl, token]);

  if (!costEstimate) return null;

  if (costEstimate.status !== "estimated") {
    return (
      <section className="estimate" aria-label="Repair cost estimate">
        <h3>Estimated repair cost</h3>
        <p className="muted">
          A cost estimate could not be produced for this claim
          {costEstimate.reason ? `: ${costEstimate.reason}` : "."} Your damage
          report above is still valid.
        </p>
        <p className="disclaimer">{DISCLAIMER}</p>
      </section>
    );
  }

  const visible = costEstimate.items ?? [];
  const internal = internalEstimate?.items ?? [];
  const sources = Array.from(
    new Map(
      visible
        .flatMap((item) => item.sources)
        .filter((source) => /^https?:\/\//i.test(source.url))
        .map((source) => [source.url, source]),
    ).values(),
  );
  const selectedTemplate = templates.find((t) => t.id === form.template);

  async function applyPolicy(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const body = {
        template: form.template,
        idv_inr: form.idv_inr ? Number(form.idv_inr) : null,
        vehicle_age_months: Number(form.vehicle_age_months) || 0,
        engine_cc: Number(form.engine_cc) || 1200,
        voluntary_deductible_inr: Number(form.voluntary_deductible_inr) || 0,
        copay_pct: form.copay_pct ? Number(form.copay_pct) : null,
      };
      const response = await fetch(`${apiBaseUrl}/api/v1/claims/${claimId}/policy`, {
        method: "PUT",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(
          typeof detail.detail === "string"
            ? detail.detail
            : "Please check the policy values and try again.",
        );
      }
      const updated = (await response.json()) as { claim_breakdown?: ClaimBreakdown };
      if (updated.claim_breakdown) onBreakdownChange(updated.claim_breakdown);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not update the estimate.");
    } finally {
      setBusy(false);
    }
  }

  const update = (key: keyof typeof form) => (value: string) =>
    setForm((current) => ({ ...current, [key]: value }));

  return (
    <section className="estimate" aria-label="Repair cost estimate">
      <p className="disclaimer" role="note">
        {DISCLAIMER} Prices come from web search for demonstration and are
        always shown as ranges.
      </p>

      <h3>Estimated repair cost (visible damage)</h3>
      {costEstimate.vehicle && costEstimate.vehicle !== "unknown" ? (
        <p className="muted">Vehicle (visual estimate): {costEstimate.vehicle}</p>
      ) : null}
      {visible.length ? (
        <>
          <ItemList items={visible} />
          <div className="range-card">
            <span className="muted">Total range</span>
            <strong>{formatRange(costEstimate.total)}</strong>
          </div>
        </>
      ) : (
        <p className="muted">No visible damage items were priced.</p>
      )}

      <h3>Estimated claimable amount</h3>
      <form className="policy-form" onSubmit={applyPolicy} aria-busy={busy}>
        <label>
          Policy type
          <select value={form.template} onChange={(e) => update("template")(e.target.value)}>
            {(templates.length
              ? templates
              : [{ id: "standard_comprehensive", label: "Standard comprehensive", description: "" }]
            ).map((t) => (
              <option key={t.id} value={t.id}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          IDV (₹, optional)
          <input type="number" min="1" inputMode="numeric" value={form.idv_inr}
            onChange={(e) => update("idv_inr")(e.target.value)} />
        </label>
        <label>
          Vehicle age (months)
          <input type="number" min="0" max="600" value={form.vehicle_age_months}
            onChange={(e) => update("vehicle_age_months")(e.target.value)} />
        </label>
        <label>
          Engine size (cc)
          <input type="number" min="1" max="10000" value={form.engine_cc}
            onChange={(e) => update("engine_cc")(e.target.value)} />
        </label>
        <label>
          Voluntary deductible (₹)
          <input type="number" min="0" value={form.voluntary_deductible_inr}
            onChange={(e) => update("voluntary_deductible_inr")(e.target.value)} />
        </label>
        <label>
          Co-pay (%)
          <input type="number" min="0" max="100" value={form.copay_pct}
            onChange={(e) => update("copay_pct")(e.target.value)} />
        </label>
        <button className="button" disabled={busy}>
          {busy ? "Updating..." : "Update claimable range"}
        </button>
      </form>
      {selectedTemplate?.description ? (
        <p className="muted">{selectedTemplate.description}</p>
      ) : null}
      {error ? <p className="error" role="alert">{error}</p> : null}

      {breakdown ? (
        <div className="range-card highlight">
          <span className="muted">
            {breakdown.covered ? "Estimated claimable range" : "Not covered by this policy type"}
          </span>
          <strong>{formatRange(breakdown.claimable)}</strong>
          <span className="muted">
            Deductible {inr(breakdown.deductible_inr)} · Depreciation {formatRange(breakdown.depreciation)}
          </span>
          {breakdown.possible_total_loss ? (
            <span className="error">
              Repair cost may approach the total-loss threshold; the insurer may assess a total loss.
            </span>
          ) : null}
        </div>
      ) : (
        <p className="muted">No claimable range could be calculated from the priced items.</p>
      )}

      {internal.length ? (
        <div className="internal-panel">
          <h3>Possible internal or hidden damage (not confirmed)</h3>
          <p className="muted">
            {internalEstimate?.note ??
              "These are possibilities only and are not included in the claimable amount."}
          </p>
          <ItemList items={internal} />
          <div className="range-card">
            <span className="muted">Additional cost if all were confirmed</span>
            <strong>{formatRange(internalEstimate?.total)}</strong>
          </div>
        </div>
      ) : null}

      <details className="assumptions">
        <summary>Assumptions and sources</summary>
        <ul>
          {[...(costEstimate.assumptions ?? []), ...(breakdown?.assumptions ?? [])].map(
            (assumption, index) => (
              <li key={index}>{assumption}</li>
            ),
          )}
        </ul>
        {costEstimate.generated_at ? (
          <p className="muted">
            Prices retrieved {costEstimate.generated_at.slice(0, 10)}
            {costEstimate.provider ? ` via ${costEstimate.provider}` : ""}.
          </p>
        ) : null}
        {sources.length ? (
          <ul>
            {sources.slice(0, 10).map((source) => (
              <li key={source.url}>
                <a href={source.url} target="_blank" rel="noopener noreferrer">
                  {source.title || source.url}
                </a>
              </li>
            ))}
          </ul>
        ) : null}
      </details>
    </section>
  );
}
