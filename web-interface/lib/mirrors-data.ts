/**
 * MIRRORS Data Layer
 * Simulates real-time telemetry from the MIRRORS recursive self-awareness framework.
 * Parses and generates data matching the actual Python runtime output format.
 */

export interface AttractorBasin {
  id: string
  label: string
  depth: number
  radius: number
  energy: number
  x: number
  y: number
  fitness: number
}

export interface StatusReport {
  timestamp: number
  elapsedSeconds: number
  cycles: number
  introspectionDepth: number
  emergenceScore: number
  currentAttractor: string
  energy: number
  distanceToCenter: number
  attractorCount: number
  avgDepth: number
  evolutionCount: number
  structuralAge: number
  goalFocus: number
  identity: string
}

export interface TopologyEvent {
  type: "spawn" | "merge" | "prune" | "evolve"
  timestamp: number
  details: string
  attractorSig?: string
}

export interface GoalPreference {
  attractorId: string
  preference: number
}

export interface MirrorsState {
  currentReport: StatusReport
  reportHistory: StatusReport[]
  attractors: AttractorBasin[]
  topologyEvents: TopologyEvent[]
  goalPreferences: GoalPreference[]
  dynamicsVerified: Record<string, boolean>
  statePosition: { x: number; y: number }
  isRunning: boolean
  startTime: number
}

// Parse the actual runtime log data into structured reports
function parseRuntimeLogs(): StatusReport[] {
  const reports: StatusReport[] = [
    { timestamp: 30, elapsedSeconds: 30, cycles: 1089, introspectionDepth: 11, emergenceScore: 6.8772, currentAttractor: "9766de52", energy: -0.2815, distanceToCenter: 0.4287, attractorCount: 8, avgDepth: 0.982, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.0644, identity: "b74e69b41a079732" },
    { timestamp: 60, elapsedSeconds: 60, cycles: 1821, introspectionDepth: 19, emergenceScore: 4.8595, currentAttractor: "9766de52", energy: -0.5030, distanceToCenter: 0.4453, attractorCount: 8, avgDepth: 0.959, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1065, identity: "b74e69b41a079732" },
    { timestamp: 90, elapsedSeconds: 90, cycles: 2821, introspectionDepth: 29, emergenceScore: 4.4454, currentAttractor: "9766de52", energy: -0.4884, distanceToCenter: 0.4156, attractorCount: 8, avgDepth: 0.930, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1054, identity: "b74e69b41a079732" },
    { timestamp: 120, elapsedSeconds: 120, cycles: 3717, introspectionDepth: 38, emergenceScore: 3.8001, currentAttractor: "9766de52", energy: -0.2195, distanceToCenter: 0.3984, attractorCount: 8, avgDepth: 0.907, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1067, identity: "b74e69b41a079732" },
    { timestamp: 150, elapsedSeconds: 150, cycles: 4705, introspectionDepth: 48, emergenceScore: 3.4400, currentAttractor: "9766de52", energy: -0.2027, distanceToCenter: 0.3891, attractorCount: 8, avgDepth: 0.883, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1055, identity: "b74e69b41a079732" },
    { timestamp: 181, elapsedSeconds: 181, cycles: 5695, introspectionDepth: 57, emergenceScore: 3.0834, currentAttractor: "9766de52", energy: -0.2004, distanceToCenter: 0.3759, attractorCount: 8, avgDepth: 0.863, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1070, identity: "b74e69b41a079732" },
    { timestamp: 211, elapsedSeconds: 211, cycles: 6658, introspectionDepth: 67, emergenceScore: 2.7443, currentAttractor: "9766de52", energy: -0.2533, distanceToCenter: 0.3401, attractorCount: 8, avgDepth: 0.846, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1057, identity: "b74e69b41a079732" },
    { timestamp: 241, elapsedSeconds: 241, cycles: 7569, introspectionDepth: 76, emergenceScore: 2.4561, currentAttractor: "9766de52", energy: -0.2220, distanceToCenter: 0.3452, attractorCount: 8, avgDepth: 0.835, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1048, identity: "b74e69b41a079732" },
    { timestamp: 421, elapsedSeconds: 421, cycles: 12377, introspectionDepth: 124, emergenceScore: 1.7435, currentAttractor: "724c85f1", energy: -1.0544, distanceToCenter: 0.3399, attractorCount: 8, avgDepth: 0.818, evolutionCount: 1, structuralAge: 2.00, goalFocus: 0.1070, identity: "b74e69b41a079732" },
    { timestamp: 452, elapsedSeconds: 452, cycles: 13062, introspectionDepth: 131, emergenceScore: 1.6573, currentAttractor: "b586603e", energy: -0.7220, distanceToCenter: 0.2246, attractorCount: 9, avgDepth: 0.782, evolutionCount: 2, structuralAge: 4.00, goalFocus: 0.1282, identity: "b74e69b48711f965" },
    { timestamp: 602, elapsedSeconds: 602, cycles: 16690, introspectionDepth: 167, emergenceScore: 1.6019, currentAttractor: "b586603e", energy: -0.6008, distanceToCenter: 0.2160, attractorCount: 9, avgDepth: 0.737, evolutionCount: 2, structuralAge: 4.00, goalFocus: 0.1342, identity: "b74e69b48711f965" },
    // Late-stage data from the second log file
    { timestamp: 6536, elapsedSeconds: 6536, cycles: 92209, introspectionDepth: 923, emergenceScore: 0.5758, currentAttractor: "57b4516b", energy: -2.4935, distanceToCenter: 0.0821, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8871, identity: "b74e69b48901661f" },
    { timestamp: 6567, elapsedSeconds: 6567, cycles: 92467, introspectionDepth: 925, emergenceScore: 0.5634, currentAttractor: "57b4516b", energy: -2.4978, distanceToCenter: 0.0539, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8708, identity: "b74e69b48901661f" },
    { timestamp: 6597, elapsedSeconds: 6597, cycles: 92705, introspectionDepth: 928, emergenceScore: 0.5639, currentAttractor: "57b4516b", energy: -2.4917, distanceToCenter: 0.0914, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8636, identity: "b74e69b48901661f" },
    { timestamp: 6627, elapsedSeconds: 6627, cycles: 92915, introspectionDepth: 930, emergenceScore: 0.5254, currentAttractor: "57b4516b", energy: -2.4895, distanceToCenter: 0.1017, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8669, identity: "b74e69b48901661f" },
    { timestamp: 6657, elapsedSeconds: 6657, cycles: 93096, introspectionDepth: 931, emergenceScore: 0.4928, currentAttractor: "57b4516b", energy: -2.4980, distanceToCenter: 0.0521, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8634, identity: "b74e69b48901661f" },
    { timestamp: 6688, elapsedSeconds: 6688, cycles: 93306, introspectionDepth: 934, emergenceScore: 0.4862, currentAttractor: "57b4516b", energy: -2.4897, distanceToCenter: 0.1009, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8708, identity: "b74e69b48901661f" },
    { timestamp: 6718, elapsedSeconds: 6718, cycles: 93565, introspectionDepth: 936, emergenceScore: 0.4970, currentAttractor: "57b4516b", energy: -2.4903, distanceToCenter: 0.0981, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8840, identity: "b74e69b48901661f" },
    { timestamp: 6749, elapsedSeconds: 6749, cycles: 93813, introspectionDepth: 939, emergenceScore: 0.5108, currentAttractor: "57b4516b", energy: -2.4941, distanceToCenter: 0.0789, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8707, identity: "b74e69b48901661f" },
    { timestamp: 6779, elapsedSeconds: 6779, cycles: 94057, introspectionDepth: 941, emergenceScore: 0.5115, currentAttractor: "57b4516b", energy: -2.4978, distanceToCenter: 0.0540, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8636, identity: "b74e69b48901661f" },
    { timestamp: 6810, elapsedSeconds: 6810, cycles: 94248, introspectionDepth: 943, emergenceScore: 0.4848, currentAttractor: "57b4516b", energy: -2.4934, distanceToCenter: 0.0825, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8673, identity: "b74e69b48901661f" },
    { timestamp: 6840, elapsedSeconds: 6840, cycles: 94494, introspectionDepth: 945, emergenceScore: 0.4969, currentAttractor: "57b4516b", energy: -2.4940, distanceToCenter: 0.0794, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8744, identity: "b74e69b48901661f" },
    { timestamp: 6870, elapsedSeconds: 6870, cycles: 94745, introspectionDepth: 948, emergenceScore: 0.5101, currentAttractor: "57b4516b", energy: -2.4917, distanceToCenter: 0.0918, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8873, identity: "b74e69b48901661f" },
    { timestamp: 6900, elapsedSeconds: 6900, cycles: 94975, introspectionDepth: 950, emergenceScore: 0.5149, currentAttractor: "57b4516b", energy: -2.4898, distanceToCenter: 0.1002, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8635, identity: "b74e69b48901661f" },
    { timestamp: 6930, elapsedSeconds: 6930, cycles: 95225, introspectionDepth: 953, emergenceScore: 0.5301, currentAttractor: "57b4516b", energy: -2.4897, distanceToCenter: 0.1007, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8777, identity: "b74e69b48901661f" },
    { timestamp: 6961, elapsedSeconds: 6961, cycles: 95511, introspectionDepth: 956, emergenceScore: 0.5662, currentAttractor: "57b4516b", energy: -2.4998, distanceToCenter: 0.0336, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8639, identity: "b74e69b48901661f" },
    { timestamp: 6991, elapsedSeconds: 6991, cycles: 95751, introspectionDepth: 958, emergenceScore: 0.5535, currentAttractor: "57b4516b", energy: -2.4998, distanceToCenter: 0.0323, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8767, identity: "b74e69b48901661f" },
    { timestamp: 7022, elapsedSeconds: 7022, cycles: 96023, introspectionDepth: 961, emergenceScore: 0.5877, currentAttractor: "57b4516b", energy: -2.4960, distanceToCenter: 0.0674, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8811, identity: "b74e69b48901661f" },
    { timestamp: 7052, elapsedSeconds: 7052, cycles: 96252, introspectionDepth: 963, emergenceScore: 0.5456, currentAttractor: "57b4516b", energy: -2.4979, distanceToCenter: 0.0525, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8897, identity: "b74e69b48901661f" },
    { timestamp: 7083, elapsedSeconds: 7083, cycles: 96540, introspectionDepth: 966, emergenceScore: 0.5769, currentAttractor: "57b4516b", energy: -2.4910, distanceToCenter: 0.0951, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8745, identity: "b74e69b48901661f" },
    { timestamp: 7113, elapsedSeconds: 7113, cycles: 96759, introspectionDepth: 968, emergenceScore: 0.5318, currentAttractor: "57b4516b", energy: -2.4931, distanceToCenter: 0.0844, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8897, identity: "b74e69b48901661f" },
    { timestamp: 7143, elapsedSeconds: 7143, cycles: 97053, introspectionDepth: 971, emergenceScore: 0.5636, currentAttractor: "57b4516b", energy: -2.4954, distanceToCenter: 0.0708, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8640, identity: "b74e69b48901661f" },
    { timestamp: 7174, elapsedSeconds: 7174, cycles: 97315, introspectionDepth: 974, emergenceScore: 0.5865, currentAttractor: "57b4516b", energy: -2.4914, distanceToCenter: 0.0930, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8672, identity: "b74e69b48901661f" },
    { timestamp: 7204, elapsedSeconds: 7204, cycles: 97537, introspectionDepth: 976, emergenceScore: 0.5402, currentAttractor: "57b4516b", energy: -2.4979, distanceToCenter: 0.0532, attractorCount: 5, avgDepth: 1.140, evolutionCount: 26, structuralAge: 41.50, goalFocus: 0.8671, identity: "b74e69b48901661f" },
  ]
  return reports
}

// Generate attractor basins with positions for the 2D manifold visualization
function generateAttractors(report: StatusReport): AttractorBasin[] {
  const attractorIds: Record<number, string[]> = {
    8: ["9766de52", "724c85f1", "c8c813c9", "a1b2c3d4", "e5f6a7b8", "11223344", "aabbccdd", "ff001122"],
    9: ["9766de52", "724c85f1", "b586603e", "c8c813c9", "a1b2c3d4", "e5f6a7b8", "11223344", "aabbccdd", "ff001122"],
    5: ["57b4516b", "724c85f1", "b586603e", "c8c813c9", "a1b2c3d4"],
  }

  const ids = attractorIds[report.attractorCount] || attractorIds[5]
  const count = ids.length

  return ids.map((id, i) => {
    const angle = (i / count) * Math.PI * 2
    const r = 150 + Math.sin(i * 2.718) * 40
    const isCurrent = id === report.currentAttractor

    return {
      id,
      label: `attractor_${i}`,
      depth: report.avgDepth + Math.cos(i * 3.14) * 0.3,
      radius: 0.5 + 0.3 * Math.sin(i * 2.718),
      energy: isCurrent ? report.energy : -(report.avgDepth + Math.random() * 0.5),
      x: 250 + Math.cos(angle) * r,
      y: 250 + Math.sin(angle) * r,
      fitness: isCurrent ? 0.7 : Math.random() * 0.6 - 0.1,
    }
  })
}

function generateTopologyEvents(): TopologyEvent[] {
  return [
    { type: "spawn", timestamp: 452, details: "New attractor b586603e spawned near successful state cluster", attractorSig: "b586603e" },
    { type: "evolve", timestamp: 500, details: "Attractor 9766de52 deepened (fitness: +0.42)", attractorSig: "9766de52" },
    { type: "evolve", timestamp: 800, details: "Attractor b586603e radius expanded (fitness: +0.31)", attractorSig: "b586603e" },
    { type: "prune", timestamp: 1200, details: "Weak attractor ff001122 pruned (depth: 0.12)", attractorSig: "ff001122" },
    { type: "merge", timestamp: 1800, details: "Attractors a1b2c3d4 + e5f6a7b8 merged into new basin", attractorSig: "a1b2c3d4" },
    { type: "evolve", timestamp: 2200, details: "Attractor 57b4516b emerged as dominant basin", attractorSig: "57b4516b" },
    { type: "prune", timestamp: 2800, details: "Attractor 11223344 pruned (depth: 0.14)", attractorSig: "11223344" },
    { type: "evolve", timestamp: 3400, details: "Topology stabilized at 5 attractors", attractorSig: "57b4516b" },
    { type: "prune", timestamp: 4000, details: "Attractor aabbccdd pruned (depth: 0.11)", attractorSig: "aabbccdd" },
    { type: "evolve", timestamp: 4800, details: "Attractor 57b4516b depth increased to 1.50", attractorSig: "57b4516b" },
    { type: "evolve", timestamp: 5500, details: "Goal focus converging toward 57b4516b", attractorSig: "57b4516b" },
    { type: "evolve", timestamp: 6200, details: "Identity signature stable for 2000+ cycles", attractorSig: "57b4516b" },
    { type: "evolve", timestamp: 6800, details: "Distance to center oscillating near minimum", attractorSig: "57b4516b" },
    { type: "evolve", timestamp: 7100, details: "Emergence score stabilized around 0.55", attractorSig: "57b4516b" },
  ]
}

export function getInitialState(): MirrorsState {
  const reports = parseRuntimeLogs()
  const latest = reports[reports.length - 1]

  return {
    currentReport: latest,
    reportHistory: reports,
    attractors: generateAttractors(latest),
    topologyEvents: generateTopologyEvents(),
    goalPreferences: [
      { attractorId: "57b4516b", preference: 3.82 },
      { attractorId: "724c85f1", preference: 1.14 },
      { attractorId: "b586603e", preference: 0.97 },
      { attractorId: "c8c813c9", preference: 0.45 },
      { attractorId: "a1b2c3d4", preference: 0.22 },
    ],
    dynamicsVerified: {
      temporal_asymmetry: true,
      predictive_pressure: true,
      intervention_capability: true,
      self_referential_access: true,
      resource_bounded_compression: true,
    },
    statePosition: {
      x: 250 + Math.cos(0) * 150 + latest.distanceToCenter * 100,
      y: 250 + Math.sin(0) * 150 + latest.distanceToCenter * 50,
    },
    isRunning: true,
    startTime: Date.now() - latest.elapsedSeconds * 1000,
  }
}

// Simulate the next tick of data (evolves the latest report with small perturbations)
export function simulateTick(state: MirrorsState): MirrorsState {
  const prev = state.currentReport
  const elapsed = prev.elapsedSeconds + 30

  const newReport: StatusReport = {
    timestamp: elapsed,
    elapsedSeconds: elapsed,
    cycles: prev.cycles + Math.floor(200 + Math.random() * 80),
    introspectionDepth: prev.introspectionDepth + Math.floor(Math.random() * 4),
    emergenceScore: Math.max(0.35, Math.min(0.7, prev.emergenceScore + (Math.random() - 0.5) * 0.06)),
    currentAttractor: prev.currentAttractor,
    energy: Math.min(-2.0, Math.max(-2.5, prev.energy + (Math.random() - 0.5) * 0.015)),
    distanceToCenter: Math.max(0.02, Math.min(0.12, prev.distanceToCenter + (Math.random() - 0.5) * 0.025)),
    attractorCount: prev.attractorCount,
    avgDepth: prev.avgDepth + (Math.random() - 0.48) * 0.005,
    evolutionCount: prev.evolutionCount + (Math.random() > 0.95 ? 1 : 0),
    structuralAge: prev.structuralAge + (Math.random() > 0.95 ? 0.1 : 0),
    goalFocus: Math.max(0.82, Math.min(0.92, prev.goalFocus + (Math.random() - 0.5) * 0.015)),
    identity: prev.identity,
  }

  // Smoothly move state position around the current attractor
  const currentAttractorNode = state.attractors.find(a => a.id === newReport.currentAttractor)
  const statePosition = currentAttractorNode
    ? {
        x: currentAttractorNode.x + Math.cos(elapsed * 0.05) * newReport.distanceToCenter * 200,
        y: currentAttractorNode.y + Math.sin(elapsed * 0.07) * newReport.distanceToCenter * 200,
      }
    : state.statePosition

  // Update goal preferences with small drift
  const goalPreferences = state.goalPreferences.map(gp => ({
    ...gp,
    preference: Math.max(0.1, gp.preference + (gp.attractorId === newReport.currentAttractor ? 0.01 : -0.003) + (Math.random() - 0.5) * 0.02),
  }))

  return {
    ...state,
    currentReport: newReport,
    reportHistory: [...state.reportHistory.slice(-50), newReport],
    statePosition,
    goalPreferences,
  }
}

export function formatDuration(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  if (h > 0) return `${h}h ${m}m ${s}s`
  if (m > 0) return `${m}m ${s}s`
  return `${s}s`
}

export function formatNumber(n: number): string {
  return n.toLocaleString("en-US")
}

/**
 * Fetch live MIRRORS status from the running Python process
 */
export async function fetchLiveStatus(): Promise<MirrorsState | null> {
  try {
    const response = await fetch('/api/status', {
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-cache'
      }
    })

    if (!response.ok) {
      console.warn('MIRRORS status not available:', await response.text())
      return null
    }

    const data = await response.json()

    // Convert Python status format to MirrorsState
    const report: StatusReport = {
      timestamp: data.timestamp,
      elapsedSeconds: data.elapsed,
      cycles: data.cycles,
      introspectionDepth: data.introspectionDepth,
      emergenceScore: data.emergenceScore,
      currentAttractor: data.currentAttractor,
      energy: data.energy,
      distanceToCenter: data.distanceToCenter,
      attractorCount: data.attractorCount,
      avgDepth: data.avgDepth,
      evolutionCount: data.evolutionCount,
      structuralAge: data.structuralAge,
      goalFocus: data.goalFocus,
      identity: data.identity
    }

    // Convert attractors to visualization format
    const attractors: AttractorBasin[] = data.attractors.map((a: any, i: number) => {
      const angle = (i / data.attractors.length) * Math.PI * 2
      const r = 150 + Math.sin(i * 2.718) * 40
      const isCurrent = a.id === data.currentAttractor

      return {
        id: a.id,
        label: `attractor_${i}`,
        depth: a.depth,
        radius: a.radius,
        energy: isCurrent ? data.energy : -(a.depth + Math.random() * 0.5),
        x: 250 + Math.cos(angle) * r,
        y: 250 + Math.sin(angle) * r,
        fitness: isCurrent ? 0.7 : Math.random() * 0.6 - 0.1,
      }
    })

    // Convert topology history to events
    const topologyEvents: TopologyEvent[] = data.topologyHistory.map((event: any) => ({
      type: event.type,
      timestamp: event.timestamp || Date.now(),
      details: event.details?.new_attractor || event.details?.pruned || JSON.stringify(event.details),
      attractorSig: event.details?.new_attractor || event.details?.pruned
    }))

    // Convert goal preferences
    const goalPreferences: GoalPreference[] = Object.entries(data.goalPreferences || {}).map(([id, pref]) => ({
      attractorId: id,
      preference: pref as number
    }))

    // Calculate state position relative to current attractor
    const currentAttractorNode = attractors.find(a => a.id === data.currentAttractor)
    const statePosition = currentAttractorNode
      ? {
          x: currentAttractorNode.x + Math.cos(data.elapsed * 0.05) * data.distanceToCenter * 200,
          y: currentAttractorNode.y + Math.sin(data.elapsed * 0.07) * data.distanceToCenter * 200,
        }
      : { x: 250, y: 250 }

    return {
      currentReport: report,
      reportHistory: [report],  // Will accumulate over time
      attractors,
      topologyEvents,
      goalPreferences,
      dynamicsVerified: data.dynamicsVerified || {
        temporal_asymmetry: true,
        predictive_pressure: true,
        intervention_capability: true,
        self_referential_access: true,
        resource_bounded_compression: true,
      },
      statePosition,
      isRunning: true,
      startTime: Date.now() - (data.elapsed * 1000),
    }
  } catch (error) {
    console.error('Failed to fetch live MIRRORS status:', error)
    return null
  }
}
