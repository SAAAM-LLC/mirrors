"use client"

import { useEffect, useState, useCallback } from "react"
import { getInitialState, simulateTick, fetchLiveStatus, formatNumber, type MirrorsState } from "@/lib/mirrors-data"
import { DashboardHeader } from "@/components/mirrors/dashboard-header"
import { ManifoldVisualizer } from "@/components/mirrors/manifold-visualizer"
import { StatCard } from "@/components/mirrors/stat-card"
import { DynamicsPanel } from "@/components/mirrors/dynamics-panel"
import { EmergenceChart, EnergyChart, GoalPreferencesChart, IntrospectionChart } from "@/components/mirrors/metrics-charts"
import { TopologyLog } from "@/components/mirrors/topology-log"
import { IdentityPanel } from "@/components/mirrors/identity-panel"

export default function MirrorsDashboard() {
  const [state, setState] = useState<MirrorsState | null>(null)
  const [useLiveData, setUseLiveData] = useState(true)
  const [liveDataAvailable, setLiveDataAvailable] = useState(false)

  // Initial fetch - try live data first, fall back to simulation
  useEffect(() => {
    async function init() {
      if (useLiveData) {
        const liveState = await fetchLiveStatus()
        if (liveState) {
          setState(liveState)
          setLiveDataAvailable(true)
          return
        }
      }
      // Fall back to simulated data
      setState(getInitialState())
      setUseLiveData(false)
      setLiveDataAvailable(false)
    }
    init()
  }, [])

  // Poll for live data updates
  useEffect(() => {
    if (!state?.isRunning || !useLiveData) return

    const interval = setInterval(async () => {
      const liveState = await fetchLiveStatus()
      if (liveState) {
        setState((prev) => {
          if (!prev) return liveState
          // Accumulate history
          return {
            ...liveState,
            reportHistory: [...prev.reportHistory.slice(-50), liveState.currentReport]
          }
        })
        setLiveDataAvailable(true)
      } else {
        setLiveDataAvailable(false)
      }
    }, 2000)

    return () => clearInterval(interval)
  }, [state?.isRunning, useLiveData])

  // Simulate data if live unavailable
  useEffect(() => {
    if (!state?.isRunning || useLiveData) return

    const interval = setInterval(() => {
      setState((prev) => {
        if (!prev) return prev
        return simulateTick(prev)
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [state?.isRunning, useLiveData])

  if (!state) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="flex flex-col items-center gap-3">
          <div className="relative">
            <div className="h-3 w-3 rounded-full bg-glow-cyan" />
            <div className="absolute inset-0 h-3 w-3 rounded-full bg-glow-cyan animate-ping opacity-75" />
          </div>
          <span className="text-xs font-mono text-muted-foreground tracking-wider uppercase">
            Initializing MIRRORS...
          </span>
        </div>
      </div>
    )
  }

  const { currentReport: r } = state

  return (
    <main className="min-h-screen bg-background text-foreground">
      <div className="mx-auto max-w-[1600px] px-4 py-6 md:px-6 lg:px-8 flex flex-col gap-6">
        {/* Header */}
        <div className="flex items-center gap-3">
          <DashboardHeader report={r} isRunning={state.isRunning} />
          {liveDataAvailable && (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <div className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="text-[10px] font-mono text-emerald-400 uppercase tracking-wider">Live</span>
            </div>
          )}
          {!liveDataAvailable && !useLiveData && (
            <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/20">
              <div className="h-2 w-2 rounded-full bg-amber-500" />
              <span className="text-[10px] font-mono text-amber-400 uppercase tracking-wider">Simulated</span>
            </div>
          )}
        </div>

        {/* Top stats row */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6">
          <StatCard
            label="Emergence"
            value={r.emergenceScore.toFixed(4)}
            subValue="normalized score"
            accentColor="cyan"
          />
          <StatCard
            label="Introspection"
            value={r.introspectionDepth.toString()}
            subValue="recursive depth"
            accentColor="cyan"
          />
          <StatCard
            label="Energy"
            value={r.energy.toFixed(4)}
            subValue="total manifold"
            accentColor="emerald"
          />
          <StatCard
            label="Dist. to Center"
            value={r.distanceToCenter.toFixed(4)}
            subValue="current basin"
            accentColor="amber"
          />
          <StatCard
            label="Goal Focus"
            value={`${(r.goalFocus * 100).toFixed(1)}%`}
            subValue="preference entropy"
            accentColor="amber"
          />
          <StatCard
            label="Cycles"
            value={formatNumber(r.cycles)}
            subValue={`${r.attractorCount} attractors`}
            accentColor="default"
          />
        </div>

        {/* Main content grid */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-12">
          {/* Left column: Manifold + Identity + Dynamics */}
          <div className="flex flex-col gap-4 lg:col-span-5">
            {/* Manifold visualization */}
            <div className="rounded-lg border border-border bg-card overflow-hidden aspect-square max-h-[500px]">
              <ManifoldVisualizer
                attractors={state.attractors}
                currentAttractor={r.currentAttractor}
                statePosition={state.statePosition}
                report={r}
              />
            </div>

            {/* Identity + Dynamics side by side */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
              <IdentityPanel report={r} />
              <DynamicsPanel dynamics={state.dynamicsVerified} />
            </div>
          </div>

          {/* Right column: Charts + Event Log */}
          <div className="flex flex-col gap-4 lg:col-span-7">
            {/* Charts grid */}
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <EmergenceChart history={state.reportHistory} />
              <EnergyChart history={state.reportHistory} />
            </div>

            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <IntrospectionChart history={state.reportHistory} />
              <GoalPreferencesChart
                preferences={state.goalPreferences}
                currentAttractor={r.currentAttractor}
              />
            </div>

            {/* Topology Log */}
            <TopologyLog events={state.topologyEvents} />
          </div>
        </div>

        {/* Footer */}
        <footer className="border-t border-border pt-4 pb-2 flex items-center justify-between text-[9px] font-mono text-muted-foreground">
          <div className="flex items-center gap-4">
            <span>SAAAM LLC</span>
            <span className="hidden sm:inline">|</span>
            <span className="hidden sm:inline">MIRRORS Framework v2.0</span>
            <span className="hidden md:inline">|</span>
            <span className="hidden md:inline">Structured Latent Manifold Architecture</span>
          </div>
          <div className="flex items-center gap-2">
            <span>Identity: {r.identity.slice(0, 8)}...{r.identity.slice(-4)}</span>
          </div>
        </footer>
      </div>
    </main>
  )
}
