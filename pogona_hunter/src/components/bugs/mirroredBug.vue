<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import holesBug from '../bugs/holesBug.vue'

export default {
  name: 'mirroredBug',
  extends: holesBug,
  props: {
    bugId: Number,
    bugsSettings: Object,
    entranceHolePos: Array,
    exitHolePos: Array,
    bugDetails: Object,
    segmentIndex: Number,
    segmentCount: Number,
    segmentBoundsOverride: Object
  },
  computed: {
    holeSize() {
      return this.bugsSettings.holeSize || [100, 100]
    },
    isRightExit() {
      return this.bugId % 2 === 1
    },
    isLeftExit() {
      return this.bugId % 2 === 0
    },
    isStaticMovement() {
      return this.bugsSettings.movementType === 'static'
    },
    screenMidX() {
      return (this.canvas && this.canvas.width / 2) || 0
    },
    segmentsCount() {
      const overrideCount = Number(this.segmentCount)
      if (Number.isFinite(overrideCount) && overrideCount > 0) {
        return overrideCount
      }
      const configuredCount = Number(this.bugsSettings && this.bugsSettings.numOfBugs)
      if (Number.isFinite(configuredCount) && configuredCount > 0) {
        return configuredCount
      }
      const fallbackCount = Array.isArray(this.bugsSettings && this.bugsSettings.bugTypes)
        ? this.bugsSettings.bugTypes.length
        : 0
      if (fallbackCount > 0) {
        return fallbackCount
      }
      return 1
    },
    segmentWidth() {
      if (this.segmentBoundsOverride &&
          Number.isFinite(this.segmentBoundsOverride.width) &&
          this.segmentBoundsOverride.width > 0) {
        return this.segmentBoundsOverride.width
      }
      const canvasWidth = this.canvas ? this.canvas.width : 0
      return this.segmentsCount > 0 ? canvasWidth / this.segmentsCount : canvasWidth
    },
    segmentBounds() {
      if (this.segmentBoundsOverride &&
          Number.isFinite(this.segmentBoundsOverride.left) &&
          Number.isFinite(this.segmentBoundsOverride.right)) {
        return this.segmentBoundsOverride
      }
      const width = this.segmentWidth || (this.canvas ? this.canvas.width : 0)
      const overrideIndex = Number(this.segmentIndex)
      const bugIndex = Number(this.bugId)
      const index = Number.isFinite(overrideIndex)
        ? overrideIndex
        : (Number.isFinite(bugIndex) ? bugIndex : 0)
      const left = width * index
      return {left, right: left + width, width}
    },
    segmentCenterX() {
      const bounds = this.segmentBounds
      if (Number.isFinite(bounds.width) && bounds.width > 0) {
        return bounds.left + (bounds.width / 2)
      }
      return this.screenMidX
    },
    useSplitCircle() {
      return this.segmentsCount > 2
    },
    circleR() {
      if (!this.useSplitCircle) {
        const radius = this.screenMidX
        return Math.abs(radius) * (this.bugsSettings.circleRadiusScale + 0.1)
      }
      const radius = this.segmentWidth / 2
      return Math.abs(radius) * (this.bugsSettings.circleRadiusScale + 0.1)
    },
    circleR0() {
      if (!this.useSplitCircle) {
        const xValue = this.xToTarget.enter[0]
        const midPointCircle = this.isLeftExit ? xValue + this.screenMidX / 4 : xValue - this.screenMidX / 4
        return [midPointCircle, this.canvas.height / 2 - 55]
      }
      return [this.segmentCenterX, this.canvas.height / 2 - 55]
    }
  },
  methods: {
    loadNextBugType() {
      this.currentBugType = this.bugsSettings.bugTypes[this.bugId]
    },
    move() {
      if (!this.isStaticMovement) {
        holesBug.methods.move.call(this)
        return
      }
      if (this.isDead || this.isRetreated || (this.isJumped && this.isJumpUpMovement)) {
        this.draw()
        return
      }
      this.frameCounter++
      if (!this.isHoleRetreatStarted && this.frameCounter > this.numFramesToRetreat) {
        this.startRetreat()
      }
      if (this.isHoleRetreatStarted) {
        this.edgeDetection()
        this.straightMove(0)
      }
      this.draw()
    },
    initiateStartPosition() {
      holesBug.methods.initiateStartPosition.call(this)

      if (this.isVerticalMovement || this.isStaticMovement) {
        this.alignToSegmentCenter()
      }

      if (this.isStaticMovement) {
        this.y = this.canvas.height / 2
        this.vx = 0
        this.vy = 0
        this.dx = 0
        this.dy = 0
      }
    },
    alignToSegmentCenter() {
      const bounds = this.segmentBounds
      const width = Number.isFinite(bounds.width) && bounds.width > 0
        ? bounds.width
        : this.segmentWidth
      const left = Number.isFinite(bounds.left) ? bounds.left : 0
      const center = left + (width / 2)
      const margin = this.currentBugSize / 2
      const minX = margin
      const maxX = this.canvas.width - margin
      this.x = Math.min(Math.max(center, minX), maxX)
    },
    edgeDetection() {
      if (this.isChangingDirection) {
        return
      }
      const radius = this.currentBugSize / 2
      const bounds = this.segmentBounds
      const leftEdge = Number.isFinite(bounds.left) ? bounds.left : 0
      const rightEdge = Number.isFinite(bounds.right) && bounds.right > leftEdge
        ? bounds.right
        : (this.canvas ? this.canvas.width : 0)
      if (this.x < leftEdge + radius || this.x > rightEdge - radius ||
          this.y < this.upper_edge || this.y > this.canvas.height - radius) {
        this.setNextAngle()
      } else if (this.frameCounter > 100 && this.isInsideHoleBoundaries()) {
        if ((this.isHoleRetreatStarted && this.isInsideExitHoleBoundaries()) || !(this.isRandomMovement || this.isRandomSpeeds)) {
          this.hideBug()
        } else {
          this.setNextAngle()
        }
      } else {
        return
      }
      this.changeDirectionTimeout()
    },
    getNotBlockedAngles() {
      const bounds = this.segmentBounds
      const leftEdge = Number.isFinite(bounds.left) ? bounds.left : 0
      const rightEdge = Number.isFinite(bounds.right) && bounds.right > leftEdge
        ? bounds.right
        : (this.canvas ? this.canvas.width : 0)
      let borderDistances = {
        top: this.y,
        bottom: this.canvas.height - this.y,
        left: this.x - leftEdge,
        right: rightEdge - this.x
      }
      let angles = []
      for (const [key, angle] of Object.entries(this.bordersAngles)) {
        if (borderDistances[key] > this.minDistFromObstacle) {
          angles.push(angle)
        }
      }
      return angles
    }
  }
}
</script>

<style scoped>
</style>
