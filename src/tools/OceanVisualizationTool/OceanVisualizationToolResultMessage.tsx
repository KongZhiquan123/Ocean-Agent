import { Box, Text } from 'ink'
import React from 'react'
import { getTheme } from '@utils/theme'
import { Out as OceanVisualizationOut } from './OceanVisualizationTool'

type Props = {
  content: OceanVisualizationOut
}

function OceanVisualizationToolResultMessage({ content }: Props): React.JSX.Element {
  const { success, output_path, stdout, stderr, exit_code } = content

  return (
    <Box flexDirection="column">
      {success && exit_code === 0 ? (
        <Box flexDirection="row">
          <Text>&nbsp;&nbsp;⎿ &nbsp;</Text>
          <Text color="green">✅ Visualization created: {output_path}</Text>
        </Box>
      ) : (
        <Box flexDirection="row">
          <Text>&nbsp;&nbsp;⎿ &nbsp;</Text>
          <Text color="red">❌ Visualization failed (exit code: {exit_code})</Text>
        </Box>
      )}

      {stdout && (
        <Box flexDirection="column" marginTop={1}>
          <Box flexDirection="row">
            <Text>&nbsp;&nbsp;⎿ &nbsp;</Text>
            <Text color={getTheme().secondaryText}>Output:</Text>
          </Box>
          <Box flexDirection="row" paddingLeft={4}>
            <Text>{stdout}</Text>
          </Box>
        </Box>
      )}

      {stderr && (
        <Box flexDirection="column" marginTop={1}>
          <Box flexDirection="row">
            <Text>&nbsp;&nbsp;⎿ &nbsp;</Text>
            <Text color="yellow">Errors:</Text>
          </Box>
          <Box flexDirection="row" paddingLeft={4}>
            <Text color="yellow">{stderr}</Text>
          </Box>
        </Box>
      )}
    </Box>
  )
}

export default OceanVisualizationToolResultMessage
