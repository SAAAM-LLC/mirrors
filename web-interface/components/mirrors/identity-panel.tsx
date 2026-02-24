"use client"

import type { StatusReport } from "@/lib/mirrors-data"

interface IdentityPanelProps {
  report: StatusReport
}

export function IdentityPanel({ report }: IdentityPanelProps) {
  // Split identity into prefix (stable) and suffix (evolving)
  const stablePrefix = report.identity.slice(0, 8)
  const evolvingSuffix = report.identity.slice(8)

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-3">
        Identity Continuity
      </h3>
      <div className="flex flex-col gap-3">
        {/* Signature display */}
        <div className="flex flex-col gap-1">
          <span className="text-[9px] font-mono text-muted-foreground">Signature</span>
          <div className="flex items-center gap-0.5 font-mono text-sm">
            <span className="text-glow-cyan">{stablePrefix}</span>
            <span className="text-glow-amber">{evolvingSuffix}</span>
          </div>
          <div className="flex items-center gap-2 text-[8px] font-mono text-muted-foreground">
            <span className="text-glow-cyan">stable core</span>
            <span>|</span>
            <span className="text-glow-amber">evolving structure</span>
          </div>
        </div>

        {/* Evolution metrics */}
        <div className="grid grid-cols-2 gap-2">
          <div className="flex flex-col gap-0.5 rounded border border-border/50 bg-secondary/30 p-2">
            <span className="text-[8px] font-mono uppercase text-muted-foreground">Evolution Count</span>
            <span className="text-sm font-mono font-semibold text-foreground">{report.evolutionCount}</span>
          </div>
          <div className="flex flex-col gap-0.5 rounded border border-border/50 bg-secondary/30 p-2">
            <span className="text-[8px] font-mono uppercase text-muted-foreground">Structural Age</span>
            <span className="text-sm font-mono font-semibold text-foreground">{report.structuralAge.toFixed(1)}</span>
          </div>
        </div>

        {/* Attractor topology summary */}
        <div className="flex items-center justify-between rounded border border-border/50 bg-secondary/30 p-2">
          <div className="flex flex-col gap-0.5">
            <span className="text-[8px] font-mono uppercase text-muted-foreground">Attractor Basins</span>
            <span className="text-sm font-mono font-semibold text-foreground">{report.attractorCount}</span>
          </div>
          <div className="flex flex-col gap-0.5 text-right">
            <span className="text-[8px] font-mono uppercase text-muted-foreground">Avg Depth</span>
            <span className="text-sm font-mono font-semibold text-foreground">{report.avgDepth.toFixed(3)}</span>
          </div>
        </div>

        {/* Stability indicator */}
        <div className="flex flex-col gap-1">
          <div className="flex items-center justify-between">
            <span className="text-[8px] font-mono uppercase text-muted-foreground">Goal Convergence</span>
            <span className="text-[9px] font-mono text-glow-amber">{(report.goalFocus * 100).toFixed(1)}%</span>
          </div>
          <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-1000"
              style={{
                width: `${report.goalFocus * 100}%`,
                background: "linear-gradient(90deg, #22d3ee, #f59e0b)",
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
