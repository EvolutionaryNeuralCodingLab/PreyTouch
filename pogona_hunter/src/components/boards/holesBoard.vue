<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;" v-on:mousedown="analyzeScreenTouch">
    <div id="bugs-board">
      <audio ref="audio1">
        <source src="../../assets/sounds/2.mp3" type="audio/mpeg">
      </audio>
      <canvas id="backgroundCanvas" v-bind:style="{background: bugsSettings.backgroundColor}"
              v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"></canvas>
      <canvas id="bugCanvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
              v-on:mousedown="setCanvasClick($event)">
        <component
          :is="bugComponent"
          v-for="(value, index) in bugsProps"
          :key="index"
          :bug-id="index"
          :bugsSettings="bugsSettings"
          :exit-hole-pos="exitHolePos(index)"
          :entrance-hole-pos="entranceHolePos(index)"
          v-bind="value"
          ref="bugChild"
          v-on:bugRetreated="endTrial"
          v-on:bugHit="$emit('bugHit', $event)"
        />
      </canvas>
    </div>
  </div>
</template>

<script>
import holesBug from '../bugs/holesBug.vue'
import mirroredBug from '../bugs/mirroredBug.vue'
import boardsMixin from './boardsMixin'

export default {
  name: 'holesBoard',
  mixins: [boardsMixin],
  components: {holesBug, mirroredBug},
  data() {
    return {
      bugsSettings: { // extends the mixin's bugSettings
        holeSize: [200, 200],
        exitHole: 'right',
        entranceHole: null,
        holesHeightScale: 0.1,
        circleHeightScale: 0.5,
        circleRadiusScale: 0.2
      },
      xpad: 100 // padding for holes
    }
  },
  computed: {
    isSplitBugsView: function () {
      return this.bugsSettings.isSplitBugsView
    },
    bugComponent: function () {
      if (this.isSplitBugsView) {
        return 'mirroredBug'
      }
      return 'holesBug'
    },
    holesPositions: function () {
      let [canvasW, canvasH] = [this.canvas.width, this.canvas.height]
      let [holeW, holeH] = this.bugsSettings.holeSize
      let configuredHolesHeight = (canvasH - holeH / 2) * this.bugsSettings.holesHeightScale
      return {
        left: [this.xpad, canvasH - holeH - configuredHolesHeight],
        right: [canvasW - holeW - this.xpad, canvasH - holeH - configuredHolesHeight]
      }
    },
    mirrorBugsProps() {
      // iterate over the bugs and assign entrance and exit holes left for the even index and right for the odd index
      const sides = this.bugsSettings.bugTypes.map((bug, i) => {
        if (i % 2 === 0) {
          return {
            entranceHole: 'left',
            exitHole: 'left',
            bugId: `${bug}_${i}`
          }
        } else {
          return {
            entranceHole: 'right',
            exitHole: 'right',
            bugId: `${bug}_${i}`
          }
        }
      })
      console.log(
        'mirrorBugsProps',
        sides)
      return sides
    }
  },
  methods: {
    initDrawing() {
      let image = new Image()
      let canvas = document.getElementById('backgroundCanvas')
      let ctx = canvas.getContext('2d')
      let [holeW, holeH] = this.bugsSettings.holeSize
      // let that = this
      image.src = require('@/assets/hole2.png')

      image.onload = () => {
        Object.values(this.holesPositions).forEach(pos => {
          ctx.drawImage(image, pos[0], pos[1], holeW, holeH)
          console.log('Drawing holes', pos[0], pos[1], holeW, holeH)
        })
      }
    },
    exitHolePos: function (bugId) {
      if (this.isSplitBugsView) {
        const exit = this.mirrorBugsProps[bugId].exitHole
        return this.holesPositions[exit]
      }
      const exitHole = this.bugsSettings.exitHole
      return this.holesPositions[exitHole]
    },
    entranceHolePos: function (bugId) {
      if (this.isSplitBugsView) {
        const entrance = this.mirrorBugsProps[bugId].entranceHole
        return this.holesPositions[entrance]
      }
      let entranceHole = this.bugsSettings.exitHole === 'left' ? 'right' : 'left'
      return this.holesPositions[entranceHole]
    },
    extraTrialData: function () {
      let d = {
        entrance_hole_pos: this.entranceHolePos,
        exit_hole_pos: this.exitHolePos,
        canvas_size: [this.canvas.width, this.canvas.height]
      }
      let bug = this.$refs.bugChild[0]
      if (bug.isMoveInCircles) {
        d['circle_radius'] = bug.r
        d['circle_position'] = bug.r0
      }
      if (bug.isAccelerateMovement || bug.isCircleAccelerateMovement) {
        d['accelerate_multiplier'] = this.bugsSettings.accelerateMultiplier
      }
      return d
    }
  }
}
</script>

<style scoped>
#bugCanvas {
  padding: 0;
  z-index: 1;
  display: block;
  position: absolute;
  bottom: 0;
  top: auto;
}
</style>
