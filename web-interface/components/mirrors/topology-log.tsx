"use client"

import { GitBranch, GitMerge, Trash2, Zap } from "lucide-react"
import type { TopologyEvent } from "@/lib/mirrors-data"
import { formatDuration } from "@/lib/mirrors-data"

interface TopologyLogProps {
  events: TopologyEvent[]
}

const eventIcons = {
  spawn: GitBranch,
  merge: GitMerge,
  prune: Trash2,
  evolve: Zap,
}

const eventColors = {
  spawn: "text-glow-emerald",
  merge: "text-glow-cyan",
  prune: "text-destructive-foreground",
  evolve: "text-glow-amber",
}

const eventBgColors = {
  spawn: "bg-glow-emerald/10",
  merge: "bg-glow-cyan/10",
  prune: "bg-destructive/10",
  evolve: "bg-glow-amber/10",
}

export function TopologyLog({ events }: TopologyLogProps) {
  const reversed = [...events].reverse()

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-3">
        Topology Evolution Log
      </h3>
      <div className="flex flex-col gap-1.5 max-h-[320px] overflow-y-auto pr-1 scrollbar-thin">
        {reversed.map((event, i) => {
          const Icon = eventIcons[event.type]
          return (
            <div
              key={i}
              className="flex items-start gap-2 rounded-md border border-border/50 bg-secondary/30 px-2.5 py-2"
            >
              <div className={`rounded p-1 ${eventBgColors[event.type]} flex-shrink-0 mt-0.5`}>
                <Icon className={`h-3 w-3 ${eventColors[event.type]}`} />
              </div>
              <div className="flex flex-col gap-0.5 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-[9px] font-mono uppercase font-semibold ${eventColors[event.type]}`}>
                    {event.type}
                  </span>
                  <span className="text-[9px] font-mono text-muted-foreground">
                    {formatDuration(event.timestamp)}
                  </span>
                </div>
                <span className="text-[10px] font-mono text-secondary-foreground leading-relaxed">
                  {event.details}
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
