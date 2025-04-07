<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import holesBug from '../bugs/holesBug.vue'

export default {
  name: `mirroredBug`,
  extends: holesBug,
  props: {
    bugId: Number,
    bugsSettings: Object,
    entranceHolePos: Array,
    exitHolePos: Array,
    bugDetails: Object
  },
  data() {
    return {
      isChangingDirection: false,
      theta: 0,
      retreatTurningFrames: 10,
      retreatTurningProgress: 0
    }
  },
  computed: {
    viewWidth() {
      return this.canvas.width
    },
    holeSize() {
      return this.bugsSettings.holeSize || [100, 100]
    },
    isRightExit() {
      return this.bugId % 2 === 1
    },
    isLeftExit() {
      return this.bugId % 2 === 0
    },
    screenMidX() {
      return (this.canvas && this.canvas.width / 2) || 0
    },
    entranceHole() {
      return this.entranceHolePos
    },
    exitHole() {
      return this.exitHolePos
    },
    entranceDelay() {
      return this.bugId * 10 * this.bugsSettings.randomizeTiming
    },
    entranceApproachDuration() {
      return this.bugId * (Math.random() * 2000) * this.bugsSettings.randomizeTiming
    },
    leftBoundary() {
      return this.screenMidX * this.bugId
    },
    rightBoundary() {
      return this.leftBoundary + this.screenMidX
    },
    circleR() {
      const radius = this.screenMidX
      return Math.abs(radius) * (this.bugsSettings.circleRadiusScale + 0.1)
    },
    circleR0() {
        const xValue = this.xToTarget.enter[0]
        const midPointCircle = this.isLeftExit ? xValue + this.screenMidX / 4 : xValue - this.screenMidX / 4
        return [midPointCircle, this.canvas.height / 2]
    },
    xToTarget() {
      const x = this.entranceHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      const y = this.entranceHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      const xTarget = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      const yTarget = this.exitHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      console.log(this.bugId, x, y, xTarget, yTarget)
      return {'enter': [x, y], 'exit': [xTarget, yTarget]}
    },
    circleTheta() {
      return (this.isRightExit ? Math.PI / 5 : (2 * Math.PI) / 3) + (this.bugId * 0.6)
    }
  },
  methods: {
    loadNextBugType() {
      this.currentBugType = this.bugsSettings.bugTypes[this.bugId]
    },
    move() {
      if (this.isDead || this.isRetreated || (this.isJumped && this.isJumpUpMovement)) {
        this.draw()
        return
      }
      this.frameCounter++

      if (this.frameCounter < this.entranceDelay && this.bugId) {
        return
      }
      this.edgeDetection()
      this.checkHoleRetreat()
      if (this.isMoveInCircles && !this.isHoleRetreatStarted) {
        if (this.frameCounter < ((this.entranceDelay + this.entranceApproachDuration) / (1000 / 60))) {
          this.initSplitBugPosition()
          return
        } else {
          this.circularMove()
        }
      } else {
        this.straightMove()
      }
      this.draw()
    },
    initSplitBugPosition() {
        const totalSteps = this.entranceApproachDuration / (1000 / 60)
        const step = (this.frameCounter - (this.entranceDelay / (1000 / 60))) / totalSteps
        const outsideX = this.x - (this.isLeftExit ? this.holeSize / 2 : -this.holeSize / 2)
        const outsideY = this.y
        this.x = this.x || outsideX
        this.x = outsideX + (this.x - outsideX) * step
        this.y = outsideY + (this.y - outsideY) * step
    },
    edgeDetection() {
      if (this.isChangingDirection) return
      // borders
      const radius = this.isInsidePolicy ? this.currentBugSize / 2 : -this.currentBugSize
      const exceedsLeft = this.x < this.leftBoundary + radius
      const exceedsRight = this.x > this.rightBoundary - radius
      const exceedsY = this.y < radius || this.y > this.canvas.height - radius

      if (exceedsLeft || exceedsRight || exceedsY) {
        this.setAfterEdgeAngleSplitView(this.leftBoundary, this.rightBoundary, radius)
      } else if (this.frameCounter > this.numFramesToRetreat && this.isInsideHoleBoundaries()) {
        if ((this.isHoleRetreatStarted && this.isInsideExitHoleBoundaries()) || !(this.isRandomMovement || this.isRandomSpeeds)) {
          this.hideBug()
        } else {
          this.setAfterEdgeAngleSplitView(this.leftBoundary, this.rightBoundary, radius)
        }
      } else {
        return
      }
      this.changeDirectionTimeout()
    },
    setAfterEdgeAngleSplitView() {
      let nextAngle = this.directionAngle
      if (this.isOutsideEndPolicy) {
        this.hideBug()
        return
      }
      let openAngles = this.getNotBlockedAnglesSplitView()
      if (openAngles.length === 0) {
        nextAngle = (this.directionAngle + Math.PI) % (2 * Math.PI)
      } else {
        openAngles.sort()
        for (let i = 0; i < openAngles.length - 1; i++) {
          if ((openAngles[i + 1] - openAngles[i]) > Math.PI / 2) {
            openAngles[i] += 2 * Math.PI
          }
        }
        openAngles.sort()
        nextAngle = Math.random() * (openAngles[openAngles.length - 1] - openAngles[0]) + openAngles[0]
      }
      this.setNextAngle(nextAngle)
      this.changeDirectionTimeout()
    },
    getNotBlockedAnglesSplitView() {
      const angles = []
      const borderDistances = {
        top: this.y,
        bottom: this.canvas.height - this.y,
        left: this.x - this.leftBoundary,
        right: this.rightBoundary - this.x
      }
      const bordersAngles = {
        top: 3 * Math.PI / 2,
        bottom: Math.PI / 2,
        left: Math.PI,
        right: 0
      }
      for (const [key, angle] of Object.entries(bordersAngles)) {
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
