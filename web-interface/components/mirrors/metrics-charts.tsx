"use client"

import { Area, AreaChart, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Bar, BarChart, Tooltip } from "recharts"
import type { StatusReport, GoalPreference } from "@/lib/mirrors-data"

interface EmergenceChartProps {
  history: StatusReport[]
}

export function EmergenceChart({ history }: EmergenceChartProps) {
  const data = history.map((r) => ({
    time: r.elapsedSeconds,
    emergence: Number(r.emergenceScore.toFixed(4)),
    goalFocus: Number(r.goalFocus.toFixed(4)),
  }))

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          Emergence Score / Goal Focus
        </h3>
        <div className="flex items-center gap-3 text-[9px] font-mono text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="h-1.5 w-3 rounded-sm bg-glow-cyan" />
            <span>Emergence</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-1.5 w-3 rounded-sm bg-glow-amber" />
            <span>Goal Focus</span>
          </div>
        </div>
      </div>
      <div className="h-[160px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="emergenceGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="goalGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,42,0.6)" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              tickFormatter={(v) => `${Math.floor(v / 60)}m`}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
              domain={[0, "auto"]}
            />
            <Tooltip
              contentStyle={{
                background: "#0a0a10",
                border: "1px solid #1e1e2a",
                borderRadius: "6px",
                fontSize: "10px",
                fontFamily: "monospace",
                color: "#e8eaed",
              }}
            />
            <Area
              type="monotone"
              dataKey="emergence"
              stroke="#22d3ee"
              strokeWidth={1.5}
              fill="url(#emergenceGrad)"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="goalFocus"
              stroke="#f59e0b"
              strokeWidth={1.5}
              fill="url(#goalGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

interface EnergyChartProps {
  history: StatusReport[]
}

export function EnergyChart({ history }: EnergyChartProps) {
  const data = history.map((r) => ({
    time: r.elapsedSeconds,
    energy: Number(r.energy.toFixed(4)),
    distance: Number(r.distanceToCenter.toFixed(4)),
  }))

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          Energy / Distance to Center
        </h3>
        <div className="flex items-center gap-3 text-[9px] font-mono text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="h-1.5 w-3 rounded-sm bg-glow-emerald" />
            <span>Energy</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="h-1.5 w-3 rounded-sm" style={{ background: "#8b5cf6" }} />
            <span>Distance</span>
          </div>
        </div>
      </div>
      <div className="h-[160px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="energyGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="distGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,42,0.6)" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              tickFormatter={(v) => `${Math.floor(v / 60)}m`}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#0a0a10",
                border: "1px solid #1e1e2a",
                borderRadius: "6px",
                fontSize: "10px",
                fontFamily: "monospace",
                color: "#e8eaed",
              }}
            />
            <Area
              type="monotone"
              dataKey="energy"
              stroke="#10b981"
              strokeWidth={1.5}
              fill="url(#energyGrad)"
              dot={false}
            />
            <Area
              type="monotone"
              dataKey="distance"
              stroke="#8b5cf6"
              strokeWidth={1.5}
              fill="url(#distGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

interface GoalPreferencesChartProps {
  preferences: GoalPreference[]
  currentAttractor: string
}

export function GoalPreferencesChart({ preferences, currentAttractor }: GoalPreferencesChartProps) {
  const sorted = [...preferences].sort((a, b) => b.preference - a.preference)

  const data = sorted.map((p) => ({
    id: p.attractorId.slice(0, 8),
    preference: Number(p.preference.toFixed(3)),
    fill: p.attractorId === currentAttractor ? "#22d3ee" : "#1e1e2a",
    stroke: p.attractorId === currentAttractor ? "#22d3ee" : "#6b7080",
  }))

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground mb-3">
        Goal Preference Distribution
      </h3>
      <div className="h-[140px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,42,0.6)" vertical={false} />
            <XAxis
              dataKey="id"
              tick={{ fontSize: 8, fill: "#6b7080", fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#0a0a10",
                border: "1px solid #1e1e2a",
                borderRadius: "6px",
                fontSize: "10px",
                fontFamily: "monospace",
                color: "#e8eaed",
              }}
            />
            <Bar dataKey="preference" radius={[3, 3, 0, 0]}>
              {data.map((entry, index) => (
                <rect key={index} fill={entry.fill} stroke={entry.stroke} strokeWidth={1} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

interface IntrospectionChartProps {
  history: StatusReport[]
}

export function IntrospectionChart({ history }: IntrospectionChartProps) {
  const data = history.map((r) => ({
    time: r.elapsedSeconds,
    depth: r.introspectionDepth,
    cycles: r.cycles,
  }))

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
          Introspection Depth Over Time
        </h3>
      </div>
      <div className="h-[140px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="depthGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(30,30,42,0.6)" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              tickFormatter={(v) => `${Math.floor(v / 60)}m`}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 9, fill: "#6b7080", fontFamily: "monospace" }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#0a0a10",
                border: "1px solid #1e1e2a",
                borderRadius: "6px",
                fontSize: "10px",
                fontFamily: "monospace",
                color: "#e8eaed",
              }}
            />
            <Area
              type="monotone"
              dataKey="depth"
              stroke="#22d3ee"
              strokeWidth={1.5}
              fill="url(#depthGrad)"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
