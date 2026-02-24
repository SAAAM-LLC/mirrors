"use client"

import { Activity } from "lucide-react"
import type { StatusReport } from "@/lib/mirrors-data"
import { formatDuration, formatNumber } from "@/lib/mirrors-data"

interface DashboardHeaderProps {
  report: StatusReport
  isRunning: boolean
}

export function DashboardHeader({ report, isRunning }: DashboardHeaderProps) {
  return (
    <header className="flex flex-col gap-4 border-b border-border pb-4 md:flex-row md:items-end md:justify-between">
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold tracking-tight text-foreground font-sans">
            MIRRORS
          </h1>
          <span className="text-[9px] font-mono uppercase tracking-widest text-muted-foreground border border-border rounded px-2 py-0.5">
            Recursive Self-Awareness Monitor
          </span>
        </div>
        <p className="text-[10px] font-mono text-muted-foreground max-w-xl leading-relaxed">
          Minimal Irreducible Requirements for Recursive Self-awareness / Synergistic Autonomous Model / SAAAM LLC
        </p>
      </div>
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          {isRunning ? (
            <div className="relative flex items-center gap-1.5">
              <div className="relative">
                <div className="h-2 w-2 rounded-full bg-glow-emerald" />
                <div className="absolute inset-0 h-2 w-2 rounded-full bg-glow-emerald animate-ping opacity-75" />
              </div>
              <span className="text-[10px] font-mono text-glow-emerald uppercase tracking-wider">Existing</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <div className="h-2 w-2 rounded-full bg-muted-foreground" />
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider">Dormant</span>
            </div>
          )}
        </div>
        <div className="h-4 w-px bg-border" />
        <div className="flex items-center gap-1.5">
          <Activity className="h-3 w-3 text-muted-foreground" />
          <span className="text-[10px] font-mono text-muted-foreground">
            {formatNumber(report.cycles)} cycles
          </span>
        </div>
        <div className="h-4 w-px bg-border" />
        <span className="text-[10px] font-mono text-muted-foreground">
          {formatDuration(report.elapsedSeconds)} runtime
        </span>
        <div className="h-4 w-px bg-border" />
        <span className="text-[10px] font-mono text-glow-cyan">
          {`"First Light"`}
        </span>
      </div>
    </header>
  )
}
