/// Updated mirroredBoard.vue
<template>
  <holes-board :bugsProps="mirrorBugsProps"
               :bugsSettings="mirrorBugsSettings"
               :bug-details="mirrorBugsProps"
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
    }
  },
  props: ['bugsSettings'],
  computed: {
    entranceHole() {
      return ['left', 'right']
    },
    mirrorBugsProps() {
      const total = this.bugsSettings.bugTypes.length
      const half = Math.floor(total / 2)

      const leftSide = this.bugsSettings.bugTypes.slice(0, half).map((type, i) => ({
        entranceHole: 'left',
        exitHole: 'left',
        bugId: `${type}_${i}`
      }))

      const rightSide = this.bugsSettings.bugTypes.slice(half).map((type, i) => {
        const index = i + half
        return {
          entranceHole: 'right',
          exitHole: 'right',
          bugId: `${type}_${index}`
        }
      })

      const props = [...leftSide, ...rightSide]
      return props
    },
    mirrorBugsSettings() {
      return {
        ...this.bugsSettings,
        entranceHole: ['left', 'right'],
        circleRadiusScale: 0.3
      }
    }
  }
}
</script>
