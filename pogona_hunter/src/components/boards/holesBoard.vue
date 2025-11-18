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
          ref="bugChild"
          v-on:bugRetreated="endTrial"
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
      return this.bugsSettings.isSplitBugsView && this.bugsSettings.numOfBugs > 1
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
      let bugTypes = this.bugsSettings.bugTypes.length > 1
        ? this.bugsSettings.bugTypes // array of size this.bugsSettings.numOfBugs with the same bug type duplicated
        : Array(this.bugsSettings.numOfBugs).fill(this.bugsSettings.bugTypes[0])
      if (this.bugsSettings.exitHole === 'right') {
        bugTypes = [...bugTypes].reverse()
      }
      console.log('Mirror bug types', bugTypes, this.bugsSettings.numOfBugs > 1)
      // iterate over the bugs and assign entrance and exit holes left for the even index and right for the odd index
      const sides = bugTypes.map((bug, i) => ({
       entranceHole: i % 2 === 0 ? 'left' : 'right',
       exitHole: i % 2 === 0 ? 'left' : 'right',
       bugId: `${bug}_${i}`
     }))
      console.log(
        'mirrorBugsProps',
        sides)
      return sides
    }
  },
  methods: {
    initDrawing() {
      // Draw the background first
      if (this.isSplitBugsView && this.bugsSettings.bugMappedBackground) {
        this.drawSplitBackground()
      } else {
        this.drawSolidBackground()
      }

      // Flash photodiode square at trial start
      this.drawSquareForPhotoDiode()

      // Then draw the holes
      let image = new Image()
      let canvas = document.getElementById('backgroundCanvas')
      let ctx = canvas.getContext('2d')
      let [holeW, holeH] = this.bugsSettings.holeSize
      image.src = require('@/assets/hole2.png')

      image.onload = () => {
        Object.values(this.holesPositions).forEach(pos => {
          ctx.drawImage(image, pos[0], pos[1], holeW, holeH)
          console.log('Drawing holes', pos[0], pos[1], holeW, holeH)
        })
      }
    },
    exitHolePos: function (bugId) {
      let exitHole = this.bugsSettings.exitHole
      if (this.isSplitBugsView) {
        const exit = this.mirrorBugsProps[bugId]
        if (!exit) {
          console.error('No exit found for bugId', bugId, this.mirrorBugsProps)
        }
        exitHole = exit['exitHole'] || exitHole
      }

      return this.holesPositions[exitHole]
    },
    entranceHolePos: function (bugId) {
      if (this.isSplitBugsView) {
        const entrance = this.mirrorBugsProps[bugId]['entranceHole']
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

      // Add bug-mapped background information
      if (this.bugsSettings.bugMappedBackground) {
        d['bug_mapped_background'] = this.bugsSettings.bugMappedBackground
        d['is_split_background'] = this.isSplitBugsView
        if (this.isSplitBugsView && this.$refs.bugChild && this.$refs.bugChild.length >= 2) {
          d['background_colors'] = {
            left: this.getBugMappedBackgroundColor(this.$refs.bugChild[0].currentBugType),
            right: this.getBugMappedBackgroundColor(this.$refs.bugChild[1].currentBugType)
          }
        } else if (this.$refs.bugChild && this.$refs.bugChild.length > 0) {
          d['background_color'] = this.getBugMappedBackgroundColor(this.$refs.bugChild[0].currentBugType)
        }
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
#bugs-board {
  position: fixed;           /* fill the whole display                   */
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
}
#backgroundCanvas {
  position: absolute;        /* match bugCanvas so they overlap perfectly*/
  top: 0;
  left: 0;
  display: block;            /* remove inline‑block whitespace           */
}
</style>
