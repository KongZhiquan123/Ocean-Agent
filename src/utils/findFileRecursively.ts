import fs from 'fs/promises';
import path from 'path';
import type { Dirent } from 'fs';

const findFileRecursively = async (dir: string, fileName: string): Promise<string | null> => {
	const queue: string[] = [dir]
	const visited = new Set<string>()
	const ignore = new Set(['.git', 'node_modules', '.venv', 'venv'])

	while (queue.length > 0) {
		const current = queue.pop() as string
		if (visited.has(current)) continue
		visited.add(current)

		let entries: Dirent[]
		try {
			entries = await fs.readdir(current, { withFileTypes: true })
		} catch {
			continue
		}

		for (const entry of entries) {
			const fullPath = path.join(current, entry.name)

			if (entry.isFile() && entry.name === fileName) {
				return fullPath
			}

			if (entry.isDirectory() && !ignore.has(entry.name)) {
				queue.push(fullPath)
			}
		}
	}

	return null
}

export default findFileRecursively