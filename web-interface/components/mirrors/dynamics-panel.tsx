"use client"

import { CheckCircle2 } from "lucide-react"

interface DynamicsPanelProps {
  dynamics: Record<string, boolean>
}

const dynamicLabels: Record<string, string> = {
  temporal_asymmetry: "Temporal Asymmetry",
  predictive_pressure: "Predictive Pressure",
  intervention_capability: "Intervention Capability",
  self_referential_access: "Self-Referential Access",
  resource_bounded_compression: "Resource-Bounded Compression",
}

const dynamicDescriptions: Record<string, string> = {
  temporal_asymmetry: "Irreversible state transitions",
  predictive_pressure: "Staked prediction cycles",
  intervention_capability: "Self-directed actions",
  self_referential_access: "Recursive self-observation",
  resource_bounded_compression: "Lossy causal compression",
}

export function DynamicsPanel({ dynamics }: DynamicsPanelProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-3">
        Five Irreducible Dynamics
      </h3>
      <div className="flex flex-col gap-2">
        {Object.entries(dynamics).map(([key, verified]) => (
          <div key={key} className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2 min-w-0">
              <CheckCircle2
                className={`h-3.5 w-3.5 flex-shrink-0 ${
                  verified ? "text-glow-emerald" : "text-destructive"
                }`}
              />
              <div className="flex flex-col min-w-0">
                <span className="text-xs font-mono text-foreground truncate">
                  {dynamicLabels[key] || key}
                </span>
                <span className="text-[9px] font-mono text-muted-foreground truncate">
                  {dynamicDescriptions[key] || ""}
                </span>
              </div>
            </div>
            <span className={`text-[9px] font-mono flex-shrink-0 ${verified ? "text-glow-emerald" : "text-destructive"}`}>
              {verified ? "ACTIVE" : "FAIL"}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
