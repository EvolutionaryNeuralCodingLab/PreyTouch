<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import holesBug from '../bugs/holesBug.vue'
import {randomRange} from '../../js/helpers'

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
      angle: 0,
      entryAngle: null,
      retreatTurningFrames: 10,
      retreatTurningProgress: 0
    }
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
      return this.viewWidth * this.bugId
    },
    rightBoundary() {
      return this.leftBoundary + this.viewWidth
    }
  },
  methods: {
    resetState() {
      this.isRetreated = false
      this.isHoleRetreatStarted = false
      this.isCircleTrackReached = true
      this.lowHorizontalNoiseStart = (this.x + this.xTarget) / 2
      this.isNoisyPartReached = false
      this.isJumped = false
      this.frameCounter = 0
    },
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
      this.x = outsideX + (this.x - outsideX) * step
      this.y = outsideY + (this.y - outsideY) * step
    },
    isInsideHoleBoundaries() {
      return this.isInsideEntranceHoleBoundaries() || this.isInsideExitHoleBoundaries()
    },
    isInsideEntranceHoleBoundaries() {
      return this.entranceHolePos[0] <= this.x && this.x <= (this.entranceHolePos[0] + this.holeSize[0]) &&
        this.entranceHolePos[1] <= this.y && this.y <= (this.entranceHolePos[1] + this.holeSize[1])
    },
    isInsideExitHoleBoundaries() {
      let xEdge = this.xTarget - (this.holeSize[0] / 2)
      let yEdge = this.yTarget - (this.holeSize[1] / 2)
      return xEdge <= this.x && this.x <= (xEdge + this.holeSize[0]) &&
        yEdge <= this.y && this.y <= (yEdge + this.holeSize[1])
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
      console.log('setAfterEdgeAngleSplitView in mirroredBug')
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
    },
    // #TODO: remove this function and only override r0 and r
    initiateStartPosition() {
      const startHole = this.entranceHole
      const [holeW, holeH] = this.holeSize
      const x0 = Array.isArray(startHole) ? startHole[0] : startHole.x
      const y0 = Array.isArray(startHole) ? startHole[1] : startHole.y

      this.x = x0 + holeW / 2
      this.y = y0 + holeH / 2
      const exitHole = this.exitHolePos
      this.xTarget = Array.isArray(exitHole) ? exitHole[0] + holeW / 2 : exitHole.x + holeW / 2
      this.yTarget = Array.isArray(exitHole) ? exitHole[1] + holeH / 2 : exitHole.y + holeH / 2

      this.resetState()

      // Set movement parameters based on the movement type.
      switch (this.bugsSettings.movementType) {
        case `circle`:
        case `circle_accelerate`:
          // this.theta = Math.PI + (this.isRightExit ? Math.PI / 5 : (2 * Math.PI) / 3)
          // this.theta += this.bugId * (Math.PI / 5)
          const radius = this.screenMidX
          this.r = Math.abs(radius) * (this.bugsSettings.circleRadiusScale + 0.1)
          const midPointCircle = this.isLeftExit ? this.x + this.screenMidX / 4 : this.x - this.screenMidX / 4

          this.r0 = [
            midPointCircle,
            this.canvas.height / 2
          ]
          this.entryAngle = (this.isRightExit ? Math.PI / 5 : (2 * Math.PI) / 3) + (this.bugId * 0.6)
          this.angle = this.entryAngle
          break
        case `half_circle`:
          this.theta = Math.PI + Math.PI / 4
          this.r = Math.abs(this.xTarget - this.x) / 2
          this.r0 = [
            midPointCircle,
            this.y + this.r / 2.3
          ]
          break
        case 'low_horizontal':
        case 'accelerate':
        case 'jump_up':
          this.y = this.y - this.bugId * this.currentBugSize
          this.yTarget = this.yTarget - this.bugId * this.currentBugSize
          this.directionAngle = this.isRightExit ? Math.PI : 2 * Math.PI
          this.startRetreat()
          break
        default:
          this.directionAngle = randomRange(Math.PI / 4, 2 * Math.PI)
          this.setNextAngle && this.setNextAngle()
      }
    },
    circularMove() {
      // must stay here because of the new calculation of r0
      const direction = this.isCounterClockWise ? -1 : 1
      this.angle += direction * Math.abs(this.vTheta) * Math.sqrt(2) / this.r

      this.x = this.r0[0] + this.r * Math.cos(this.angle)
      this.y = this.r0[1] + this.r * Math.sin(this.angle)
    }
  }
}
</script>
<style scoped>
</style>
