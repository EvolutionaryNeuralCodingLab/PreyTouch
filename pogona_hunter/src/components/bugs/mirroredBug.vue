<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import holesBug from '../bugs/holesBug.vue'
import {distance, randomRange} from '../../js/helpers'

export default {
  name: `mirroredBug`,
  extends: holesBug,
  props: {
    bugId: Number,
    bugsSettings: Object,
    entranceHolePos: Array,
    exitHolePos: Array
  },
  data() {
    return {
      isRetreating: false,
      isChangingDirection: false
    }
  },
  computed: {
    holeSize() {
      return this.bugsSettings.holeSize || [100, 100]
    },
    isRightExit() {
      return this.bugId % 2 === 1
    },
    iLeftExit() {
      return this.bugId % 2 === 0
    },
    screenMidX() {
      return (this.canvas && this.canvas.width / 2) || 0
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
    move() {
      if (this.isDead || this.isRetreated || (this.isJumped && this.isJumpUpMovement)) {
        this.draw()
        return
      }
      this.frameCounter++

      // Check if the bug is inside any hole boundaries. If yes, end the trial.
      if (this.frameCounter > (this.numFramesToRetreat / 2) && this.isInsideHoleBoundaries()) {
        this.$emit('bugRetreated', this.bugId)
        return
      }

      this.edgeDetection()
      this.checkHoleRetreat()
      if (this.isHalfCircleMovement || (this.isMoveInCircles && !this.isHoleRetreatStarted)) {
        this.circularMove()
      } else if (this.isNoisyLowHorizontalMovement) {
        this.checkNoisyTrack()
        if (this.isNoisyPartReached) {
          this.noisyMove()
        } else {
          this.straightMove(0)
        }
      } else if (this.isLowHorizontalMovement) {
        this.straightMove(0)
      } else if (this.isJumpUpMovement || this.isAccelerateMovement) {
        this.straightMove(0)
      } else {
        this.straightMove()
      }
      this.draw()
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
      console.log('detecting')
      if (this.isChangingDirection) return

      const radius = this.isInsidePolicy ? this.currentBugSize / 2 : -this.currentBugSize
      const exceedsLeft = this.x < this.leftBoundary + radius
      const exceedsRight = this.x > this.rightBoundary - radius
      const exceedsY = this.y < radius || this.y > this.canvas.height - radius

      if (exceedsLeft || exceedsRight || exceedsY) {
        this.setAfterEdgeAngleSplitView(this.leftBoundary, this.rightBoundary, radius)
      }
    },
    setAfterEdgeAngleSplitView() {
      console.log('setAfterEdgeAngleSplitView in mirroredBug')
      let nextAngle = this.directionAngle
      if (this.isOutsideEndPolicy) {
        this.hideBug()
        return
      }
      // if (this.bugsSettings.movementType === 'low_horizontal') {
      //   // For low_horizontal bugs, always retreat toward exit with a fixed direction of -π
      //   this.directionAngle = -(2 * Math.PI)
      //   return
      // }
      let openAngles = this.getNotBlockedAnglesSplitView()
      if (openAngles.length === 0) {
        // If no open angles, default to reverse direction
        nextAngle = (this.directionAngle + Math.PI) % (2 * Math.PI)
      } else {
        openAngles.sort()
        // Maintain continuity for large gaps
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
    setNextAngle(angle = null) {
      if (this.isNoisyLowHorizontalMovement && this.isNoisyPartReached) {
        return
      }
      let nextAngle = angle
      if (!angle) {
        let openAngles = this.getNotBlockedAngles()
        openAngles = openAngles.sort()
        for (let i = 0; i < openAngles.length - 1; i++) {
          // in order to maintain the continuity in angles range, in cases of missing angles add 2π to the angles
          // right before the missing ones.
          if ((openAngles[i + 1] - openAngles[i]) > (Math.PI / 2)) {
            openAngles[i] += 2 * Math.PI
          }
        }
        openAngles = openAngles.sort()
        nextAngle = Math.random() * (openAngles[openAngles.length - 1] - openAngles[0]) + openAngles[0]
      }
      this.vx = this.currentSpeed * Math.cos(nextAngle)
      this.vy = this.currentSpeed * Math.sin(nextAngle)
    },
    initiateStartPosition() {
      // For even bug IDs use the entranceHolePos; for odd IDs use the exitHolePos as entrance.
      const startHole = (this.bugId % 2 === 0) ? this.entranceHolePos : this.exitHolePos

      // Retrieve hole width/height from settings.
      const [holeW, holeH] = this.holeSize

      // Use array indexing to pick the x and y coordinates.
      const x0 = Array.isArray(startHole) ? startHole[0] : startHole.x
      const y0 = Array.isArray(startHole) ? startHole[1] : startHole.y

      // The exit target is always given by exitHolePos.
      const xt = Array.isArray(this.exitHolePos) ? this.exitHolePos[0] : this.exitHolePos.x
      const yt = Array.isArray(this.exitHolePos) ? this.exitHolePos[1] : this.exitHolePos.y

      // Center the bug in the hole.
      this.x = x0 + holeW / 2
      this.y = y0 + holeH / 2
      this.xTarget = xt + holeW / 2
      this.yTarget = yt + holeH / 2

      console.log(`Bug`, this.bugId, `x/y:`, this.x, this.y, `targetX/targetY:`, this.xTarget, this.yTarget)
      // Reset movement-related state.
      this.resetState()
      // Set movement parameters based on the movement type.
      switch (this.bugsSettings.movementType) {
        case `circle`:
        case `circle_accelerate`:
          this.theta = Math.PI + (this.isRightSide ? Math.PI / 5 : (2 * Math.PI) / 3)
          this.theta += this.bugId * (Math.PI / 5)
          this.r = Math.abs(this.xTarget - this.x) * this.bugsSettings.circleRadiusScale
          const canvasHeight = (this.canvas && this.canvas.height) || 600
          this.r0 = [
            (this.x + this.xTarget) / 2,
            canvasHeight * this.bugsSettings.circleHeightScale
          ]
          break
        case `half_circle`:
          this.theta = Math.PI + Math.PI / 4
          this.r = Math.abs(this.xTarget - this.x) / 2
          this.r0 = [
            (this.x + this.xTarget) / 2,
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
          this.directionAngle = randomRange(3 * Math.PI / 4, 2 * Math.PI)
          this.setNextAngle && this.setNextAngle()
      }
    },
    circularMove() {
      this.theta += Math.abs(this.vTheta) * Math.sqrt(2) / this.r
      this.x = this.r0[0] + (this.r * Math.cos(this.theta)) * (this.isCounterClockWise ? -1 : 1)
      this.y = this.r0[1] + this.r * Math.sin(this.theta)
    },
    noisyMove() {
      let randNoise = this.getRandomNoise()
      this.dx = this.vx + 0.5 * randNoise
      this.dy = 0.00008 * (this.yTarget - this.y) + 0.9 * randNoise + 0.65 * this.dy
      this.x += this.dx
      this.y += this.dy
    },
    isHit(x, y) {
      if (this.isMoveInCircles && this.isHoleRetreatStarted) {
        // in case of circles don't consider hits while the bug is retreating from the circle
        return false
      }
      return distance(x, y, this.x, this.y) <= this.currentBugSize / 1.5
    },
    jump() {
      if (!(this.isJumpUpMovement || this.isAccelerateMovement || this.isCircleAccelerateMovement) ||
        this.isDead ||
        this.isJumped ||
        (this.isCircleAccelerateMovement && this.isHoleRetreatStarted)) {
        return
      }
      this.isJumped = true
      if (this.isJumpUpMovement) {
        let newY = this.y - this.jump_distance
        if (newY < this.upper_edge) {
          newY = this.upper_edge + 1
        }
        this.y = newY
      } else if (this.isAccelerateMovement) {
        this.vx = this.vx * this.bugsSettings.accelerateMultiplier
      } else if (this.isCircleAccelerateMovement) {
        this.vTheta = this.vTheta * this.bugsSettings.accelerateMultiplier
      }
      console.log('jump')
      this.jumpTimeout()
    },
    checkHoleRetreat() {
      // check if the trial duration is over and start the retreat
      if (!this.isHoleRetreatStarted && this.frameCounter > this.numFramesToRetreat) {
        this.startRetreat()
      }
    },
    startRetreat() {
      console.log('startRetreat', this.isHoleRetreatStarted)
      if (!this.isHoleRetreatStarted) {
        this.setRetreatSpeeds()
        this.isHoleRetreatStarted = true
        console.log('retreat started')
      }
    },
    setRetreatSpeeds() {
      let xd = this.xTarget - this.x
      let yd = this.yTarget - this.y
      let T = yd / xd
      // in circle movements the retreat must happen quick, otherwise use the configured bug speed
      let speed = this.isMoveInCircles ? 10 : this.currentSpeed
      this.vx = Math.sign(xd) * (speed / Math.sqrt(1 + T ** 2))
      this.vy = Math.sign(yd) * Math.sqrt((speed ** 2) - (this.vx ** 2))
    },
    checkNoisyTrack() {
      if (this.isHoleRetreatStarted) {
        return
      }
      if (!this.isNoisyPartReached) {
        if (((this.exitHolePos[0] > this.lowHorizontalNoiseStart) && (this.x > this.lowHorizontalNoiseStart)) ||
          ((this.exitHolePos[0] < this.lowHorizontalNoiseStart) && (this.x < this.lowHorizontalNoiseStart))) {
          this.isNoisyPartReached = true
        }
      }
      if (((this.exitHolePos[0] > this.lowHorizontalNoiseStart) && (this.x > this.exitHolePos[0] - 10)) ||
        ((this.exitHolePos[0] < this.lowHorizontalNoiseStart) && (this.x < this.exitHolePos[0] + 10))) {
        this.isNoisyPartReached = false
        this.startRetreat()
      }
    },
    jumpTimeout() {
      this.isJumped = true
      let t = setTimeout(() => {
        this.isJumped = false
        if (this.isJumpUpMovement) {
          this.y = this.exitHolePos[1] + (this.currentBugSize / 2)
        } else if (this.isAccelerateMovement) {
          this.vx = this.vx / this.bugsSettings.accelerateMultiplier
        } else if (this.isCircleAccelerateMovement) {
          this.vTheta = this.vTheta / this.bugsSettings.accelerateMultiplier
        }
        this.setNextAngle(this.directionAngle)
        console.log(this.vx, this.dx, this.currentSpeed)
        clearTimeout(t)
      }, 300)
    }

  }
}
</script>
<style scoped>
</style>
