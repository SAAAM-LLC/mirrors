"use client"

interface StatCardProps {
  label: string
  value: string | number
  subValue?: string
  accentColor?: "cyan" | "amber" | "emerald" | "default"
  mono?: boolean
}

const accentStyles = {
  cyan: "text-glow-cyan",
  amber: "text-glow-amber",
  emerald: "text-glow-emerald",
  default: "text-foreground",
}

export function StatCard({ label, value, subValue, accentColor = "default", mono = true }: StatCardProps) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-border bg-card p-3">
      <span className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span className={`text-lg font-semibold ${mono ? "font-mono" : "font-sans"} ${accentStyles[accentColor]} leading-none`}>
        {value}
      </span>
      {subValue && (
        <span className="text-[10px] font-mono text-muted-foreground">{subValue}</span>
      )}
    </div>
  )
}
