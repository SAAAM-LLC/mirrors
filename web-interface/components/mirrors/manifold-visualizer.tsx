"use client"

import { useEffect, useRef } from "react"
import type { AttractorBasin, StatusReport } from "@/lib/mirrors-data"

interface ManifoldVisualizerProps {
  attractors: AttractorBasin[]
  currentAttractor: string
  statePosition: { x: number; y: number }
  report: StatusReport
}

export function ManifoldVisualizer({ attractors, currentAttractor, statePosition, report }: ManifoldVisualizerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animationRef = useRef<number>(0)
  const timeRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    ctx.scale(dpr, dpr)
    const w = rect.width
    const h = rect.height

    function draw() {
      if (!ctx) return
      timeRef.current += 0.016

      ctx.clearRect(0, 0, w, h)

      // Background grid
      ctx.strokeStyle = "rgba(30, 30, 42, 0.5)"
      ctx.lineWidth = 0.5
      const gridSize = 30
      for (let x = 0; x < w; x += gridSize) {
        ctx.beginPath()
        ctx.moveTo(x, 0)
        ctx.lineTo(x, h)
        ctx.stroke()
      }
      for (let y = 0; y < h; y += gridSize) {
        ctx.beginPath()
        ctx.moveTo(0, y)
        ctx.lineTo(w, y)
        ctx.stroke()
      }

      // Scale factor to fit the 500x500 coordinate space into canvas
      const sx = w / 500
      const sy = h / 500

      // Draw saddle point connections (energy pathways between attractors)
      for (let i = 0; i < attractors.length; i++) {
        for (let j = i + 1; j < attractors.length; j++) {
          const a1 = attractors[i]
          const a2 = attractors[j]
          const dist = Math.sqrt((a1.x - a2.x) ** 2 + (a1.y - a2.y) ** 2)
          if (dist < 250) {
            const alpha = Math.max(0.04, 0.15 - dist * 0.0005)
            ctx.strokeStyle = `rgba(34, 211, 238, ${alpha})`
            ctx.lineWidth = 1
            ctx.setLineDash([4, 8])
            ctx.beginPath()
            ctx.moveTo(a1.x * sx, a1.y * sy)
            ctx.lineTo(a2.x * sx, a2.y * sy)
            ctx.stroke()
            ctx.setLineDash([])
          }
        }
      }

      // Draw attractor basins
      attractors.forEach((attractor) => {
        const ax = attractor.x * sx
        const ay = attractor.y * sy
        const isCurrent = attractor.id === currentAttractor
        const baseRadius = (attractor.radius * 60 + 20) * Math.min(sx, sy)
        const pulseRadius = baseRadius + (isCurrent ? Math.sin(timeRef.current * 2) * 4 : 0)

        // Energy well glow (outer)
        const wellGradient = ctx.createRadialGradient(ax, ay, 0, ax, ay, pulseRadius * 1.8)
        if (isCurrent) {
          wellGradient.addColorStop(0, "rgba(34, 211, 238, 0.12)")
          wellGradient.addColorStop(0.5, "rgba(34, 211, 238, 0.04)")
          wellGradient.addColorStop(1, "rgba(34, 211, 238, 0)")
        } else {
          wellGradient.addColorStop(0, "rgba(107, 112, 128, 0.08)")
          wellGradient.addColorStop(1, "rgba(107, 112, 128, 0)")
        }
        ctx.fillStyle = wellGradient
        ctx.beginPath()
        ctx.arc(ax, ay, pulseRadius * 1.8, 0, Math.PI * 2)
        ctx.fill()

        // Basin circle
        ctx.strokeStyle = isCurrent ? "rgba(34, 211, 238, 0.7)" : "rgba(107, 112, 128, 0.25)"
        ctx.lineWidth = isCurrent ? 2 : 1
        ctx.beginPath()
        ctx.arc(ax, ay, pulseRadius, 0, Math.PI * 2)
        ctx.stroke()

        // Depth indicator (filled core proportional to depth)
        const coreRadius = Math.min(pulseRadius * 0.3, attractor.depth * 8 * Math.min(sx, sy))
        const coreGradient = ctx.createRadialGradient(ax, ay, 0, ax, ay, coreRadius)
        if (isCurrent) {
          coreGradient.addColorStop(0, "rgba(34, 211, 238, 0.8)")
          coreGradient.addColorStop(1, "rgba(34, 211, 238, 0.2)")
        } else {
          coreGradient.addColorStop(0, "rgba(107, 112, 128, 0.5)")
          coreGradient.addColorStop(1, "rgba(107, 112, 128, 0.1)")
        }
        ctx.fillStyle = coreGradient
        ctx.beginPath()
        ctx.arc(ax, ay, coreRadius, 0, Math.PI * 2)
        ctx.fill()

        // Attractor ID label
        ctx.fillStyle = isCurrent ? "rgba(34, 211, 238, 0.9)" : "rgba(107, 112, 128, 0.6)"
        ctx.font = `${10 * Math.min(sx, sy)}px 'Geist Mono', monospace`
        ctx.textAlign = "center"
        ctx.fillText(attractor.id.slice(0, 8), ax, ay + pulseRadius + 14 * Math.min(sx, sy))
      })

      // Draw the current state position (the "consciousness point")
      const spx = statePosition.x * sx
      const spy = statePosition.y * sy

      // Trail effect
      const trailPoints = 8
      for (let i = 0; i < trailPoints; i++) {
        const tOffset = i * 0.15
        const tx = spx + Math.cos(timeRef.current * 1.5 - tOffset) * (i * 2)
        const ty = spy + Math.sin(timeRef.current * 1.8 - tOffset) * (i * 2)
        const alpha = 0.3 - i * 0.035
        ctx.fillStyle = `rgba(245, 158, 11, ${Math.max(0, alpha)})`
        ctx.beginPath()
        ctx.arc(tx, ty, Math.max(1, (3 - i * 0.3) * Math.min(sx, sy)), 0, Math.PI * 2)
        ctx.fill()
      }

      // State point glow
      const stateGlow = ctx.createRadialGradient(spx, spy, 0, spx, spy, 16 * Math.min(sx, sy))
      stateGlow.addColorStop(0, "rgba(245, 158, 11, 0.6)")
      stateGlow.addColorStop(0.5, "rgba(245, 158, 11, 0.15)")
      stateGlow.addColorStop(1, "rgba(245, 158, 11, 0)")
      ctx.fillStyle = stateGlow
      ctx.beginPath()
      ctx.arc(spx, spy, 16 * Math.min(sx, sy), 0, Math.PI * 2)
      ctx.fill()

      // State point core
      ctx.fillStyle = "#f59e0b"
      ctx.beginPath()
      ctx.arc(spx, spy, 4 * Math.min(sx, sy), 0, Math.PI * 2)
      ctx.fill()

      animationRef.current = requestAnimationFrame(draw)
    }

    draw()

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
    }
  }, [attractors, currentAttractor, statePosition])

  return (
    <div className="relative w-full h-full min-h-[300px]">
      <canvas
        ref={canvasRef}
        className="w-full h-full rounded-lg"
        style={{ background: "rgba(5, 5, 8, 0.8)" }}
      />
      <div className="absolute top-3 left-3 flex items-center gap-2">
        <div className="h-2 w-2 rounded-full bg-glow-cyan animate-pulse" />
        <span className="text-[10px] font-mono text-muted-foreground tracking-wider uppercase">
          Structured Latent Manifold
        </span>
      </div>
      <div className="absolute bottom-3 right-3 flex items-center gap-4 text-[10px] font-mono text-muted-foreground">
        <div className="flex items-center gap-1.5">
          <div className="h-1.5 w-1.5 rounded-full bg-glow-cyan" />
          <span>Attractor Basin</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="h-1.5 w-1.5 rounded-full bg-glow-amber" />
          <span>State Position</span>
        </div>
      </div>
    </div>
  )
}
