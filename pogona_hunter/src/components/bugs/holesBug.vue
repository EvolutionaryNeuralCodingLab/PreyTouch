<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import {randomRange} from '@/js/helpers'
import bugsMixin from './bugsMixin'
import {distance} from '../../js/helpers'

export default {
  name: 'holeBugs',
  mixins: [bugsMixin],
  data() {
    return {
      edgesPolicy: 'inside',
      framesUntilExitFromEntranceHole: 100,
      circularSpeed: null
    }
  },
  props: {
    bugId: Number,
    bugsSettings: Object,
    exitHolePos: Array,
    entranceHolePos: Array
  },
  computed: {
    holeSize: function () {
      return this.bugsSettings.holeSize
    },
    currentSpeed: function () {
      if (this.bugsSettings && this.bugsSettings.speed) {
        return this.bugsSettings.speed
      }
      return this.bugTypeOptions[this.currentBugType].speed
    },
    jump_distance: function () {
      return this.currentBugSize * 1.5
    },
    upper_edge: function () {
      let edge = this.currentBugSize / 2
      if (this.isJumpUpMovement) {
        edge += this.jump_distance
      }
      return edge
    },
    numFramesToRetreat: function () {
      return (this.bugsSettings.trialDuration || 1) * 60
    },
    isRightExit: function () {
      return this.bugsSettings.exitHole === 'right'
    },
    isLeftExit: function () {
      return this.bugsSettings.exitHole === 'left'
    },
    isMoveInCircles: function () {
      return this.bugsSettings.movementType === 'circle' || this.bugsSettings.movementType === 'circle_accelerate'
    },
    isHalfCircleMovement: function () {
      return this.bugsSettings.movementType === 'half_circle'
    },
    isRandomMovement: function () {
      return this.bugsSettings.movementType === 'random'
    },
    isLowHorizontalMovement: function () {
      return this.bugsSettings.movementType === 'low_horizontal'
    },
    isJumpUpMovement: function () {
      return this.bugsSettings.movementType === 'jump_up'
    },
    isCircleAccelerateMovement: function () {
      return this.bugsSettings.movementType === 'circle_accelerate'
    },
    isAccelerateMovement: function () {
      return this.bugsSettings.movementType === 'accelerate'
    },
    isNoisyLowHorizontalMovement: function () {
      return this.bugsSettings.movementType === 'low_horizontal_noise'
    },
    isRandomSpeeds: function () {
      return this.bugsSettings.movementType === 'random_speeds'
    },
    isCounterClockWise: function () {
      return this.isLeftExit
    },
    circleR() {
      return Math.abs(this.xTarget - this.x) * this.bugsSettings.circleRadiusScale
    },
    circleR0() {
      return [(this.x + this.xTarget) / 2, this.canvas.height * this.bugsSettings.circleHeightScale]
    },
    xToTarget() {
      const x = this.entranceHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      const y = this.entranceHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      const xTarget = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      const yTarget = this.exitHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      return {'enter': [x, y], 'exit': [xTarget, yTarget]}
    },
    circleTheta() {
      let theta = this.isRightExit ? (Math.PI + (Math.PI / 5)) : (Math.PI + (2 * Math.PI / 3))
      theta += this.bugId * (Math.PI / 5)
      return theta
    }
  },
  methods: {
    move() {
      if (this.isDead || this.isRetreated || (this.isJumped && this.isJumpUpMovement)) {
        this.draw()
        return
      }
      this.frameCounter++
      this.edgeDetection()
      this.checkHoleRetreat()
      // circle
      if (this.isHalfCircleMovement || (this.isMoveInCircles && !this.isHoleRetreatStarted)) {
        this.circularMove()
      // low horizontal noise
      } else if (this.isNoisyLowHorizontalMovement) {
        this.checkNoisyTrack()
        if (this.isNoisyPartReached) {
          this.noisyMove()
        } else {
          this.straightMove(0)
        }
      // low horizontal
      } else if (this.isLowHorizontalMovement) {
        this.straightMove(0)
      // jump up
      } else if (this.isJumpUpMovement || this.isAccelerateMovement) {
        this.straightMove(0)
      // random
      } else {
        this.straightMove()
      }
      this.draw()
    },
    edgeDetection() {
      if (this.isChangingDirection) {
        return
      }
      // borders
      let radius = this.currentBugSize / 2
      if (this.x < radius || this.x > this.canvas.width - radius ||
          this.y < this.upper_edge || this.y > this.canvas.height - radius) {
        this.setNextAngle()
      // holes edges
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
      this.x = this.xToTarget.enter[0]
      this.y = this.xToTarget.enter[1]
      this.xTarget = this.xToTarget.exit[0]
      this.yTarget = this.xToTarget.exit[1]
      this.isRetreated = false
      this.isHoleRetreatStarted = false
      this.isCircleTrackReached = true
      this.lowHorizontalNoiseStart = (this.x + this.xTarget) / 2
      this.isNoisyPartReached = false
      this.isJumped = false
      this.frameCounter = 0
      switch (this.bugsSettings.movementType) {
        case 'circle':
        case 'circle_accelerate':
          this.theta = this.circleTheta
          this.r = this.circleR
          this.r0 = this.circleR0
          break
        case 'half_circle':
          this.theta = this.isCounterClockWise ? (Math.PI + (Math.PI / 4)) : (Math.PI + (Math.PI / 4))
          this.r = (Math.abs(this.xTarget - this.x) / 2)
          this.r0 = [(this.x + this.xTarget) / 2, this.y + (this.r / 2.3)]
          break
        case 'low_horizontal':
        case 'accelerate':
        case 'jump_up':
          this.y = this.y - this.bugId * this.currentBugSize
          this.yTarget = this.yTarget - this.bugId * this.currentBugSize
          this.directionAngle = this.isRightExit ? 2 * Math.PI : Math.PI
          this.startRetreat()
          break
        case 'low_horizontal_noise':
          this.directionAngle = this.isRightExit ? 2 * Math.PI : Math.PI
          this.setRetreatSpeeds()
          break
        case 'random_speeds':
          this.directionAngle = randomRange(3 * Math.PI / 4, 2 * Math.PI)
          this.bugsSettings.speed = randomRange(2, 10)
          this.setNextAngle()
          break
        default:
          this.directionAngle = randomRange(3 * Math.PI / 4, 2 * Math.PI)
          this.setNextAngle()
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
    isInsideHoleBoundaries() {
      return this.isInsideEntranceHoleBoundaries() || this.isInsideExitHoleBoundaries()
    },
    isInsideEntranceHoleBoundaries() {
      return this.entranceHolePos[0] <= this.x && this.x <= (this.entranceHolePos[0] + this.holeSize[0]) &&
          this.entranceHolePos[1] <= this.y && this.y <= (this.entranceHolePos[1] + this.holeSize[1])
    },
    isInsideExitHoleBoundaries() {
      // this.xTarget = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      // this.yTarget = this.exitHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      let xEdge = this.xTarget - (this.bugsSettings.holeSize[0] / 2)
      let yEdge = this.yTarget - (this.bugsSettings.holeSize[1] / 2)
      return xEdge <= this.x && this.x <= (xEdge + this.holeSize[0]) &&
          yEdge <= this.y && this.y <= (yEdge + this.holeSize[1])
    },
    checkHoleRetreat() {
      // check if the trial duration is over and start the retreat
      if (!this.isHoleRetreatStarted && this.frameCounter > this.numFramesToRetreat) {
        this.startRetreat()
      }
    },
    startRetreat() {
      if (!this.isHoleRetreatStarted) {
        this.setRetreatSpeeds()
        this.isHoleRetreatStarted = true
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
