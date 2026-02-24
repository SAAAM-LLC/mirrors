import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'

export const dynamic = 'force-dynamic'
export const revalidate = 0

export async function GET() {
  try {
    const statusPath = join(process.cwd(), '.mirrors-status.json')
    const data = await readFile(statusPath, 'utf-8')
    const status = JSON.parse(data)

    return NextResponse.json(status, {
      headers: {
        'Cache-Control': 'no-store, must-revalidate',
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    })
  } catch (error) {
    // If file doesn't exist or can't be read, return placeholder
    return NextResponse.json(
      {
        error: 'MIRRORS not running or status file not found',
        message: 'Start the MIRRORS system by running: python core.py'
      },
      { status: 503 }
    )
  }
}
