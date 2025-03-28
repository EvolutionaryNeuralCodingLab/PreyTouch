/// Updated mirroredBoard.vue
<template>
  <holes-board :bugsProps="mirrorBugsProps"
               :bugsSettings="mirrorBugsSettings"
               :bug-component="bugComponent"
               :entrance-hole="mirrorBugsSettings.entranceHole"
               ref="board"/>
</template>

<script>
import holesBoard from './holesBoard.vue'
import mirroredBug from '../bugs/mirroredBug.vue'

export default {
  name: 'mirrored-board',
  components: {holesBoard},
  data() {
    return {
      bugComponent: mirroredBug
      // entranceHole: ['left', 'right']
    }
  },
  props: ['bugsSettings'],
  computed: {
    entranceHole() {
      return ['left', 'right']
    },
    mirrorBugsProps() {
      const total = this.bugsSettings.numOfBugs
      const half = Math.ceil(total / 2)
      const width = window.innerWidth
      const height = window.innerHeight

      const sideProps = (count, startX, endX, offset = 0) => {
        const spacing = (endX - startX) / count
        return Array.from({length: count}, (_, i) => ({
          x: startX + spacing * (i + 0.5),
          y: height / 2,
          bugId: `${this.bugsSettings.bugTypes[offset + i] || 'bug'}_${offset + i}`
        }))
      }

      const left = sideProps(half, 0, width / 2, 0)
      const right = sideProps(total - half, width / 2, width, half)

      return [...left, ...right]
    },
    mirrorBugsSettings() {
      return {
        ...this.bugsSettings,
        entranceHole: ['left', 'right']
      }
    }
  }
}
</script>
