<template>
  <div>
    <img ref="bugImg" :src="bugImgSrc" alt=""/>
  </div>
</template>

<script>
import bugsMixin from './bugsMixin'
import {randomRange} from '../../js/helpers'

export default {
  name: 'tunnelBug',
  mixins: [bugsMixin],
  props: {
    tunnelImagePos: Object,
    exitHolePos: Array,
    entranceHolePos: Array
  },
  data() {
    return {
      directionAngle: 0,
      theta: 0,
      r: 0,
      r0: [0, 0],
      isMoveInCircles: true,
      isCircleTrackReached: false,
      isHoleRetreatStarted: false,
      frameCounter: 0,
      isCounterClockWise: false,
      // Movement state tracking
      isMovingHorizontally: true,
      horizontalTargetX: 0,
      circleTransitionStarted: false,
      vx: 0,
      vy: 0,
      // Store original distance for radius calculation
      originalDistanceBetweenHoles: 0
      // Removed custom exit timer - using bugsMixin logic instead
    }
  },
  computed: {
    initialYs: function () {
      return [this.canvas.height / 2]
    },
    isRightExit() {
      return this.bugsSettings.exitHole === 'right'
    },
    isLeftExit() {
      return this.bugsSettings.exitHole === 'left'
    },
    currentSpeed() {
      if (this.bugsSettings && this.bugsSettings.speed) {
        return this.bugsSettings.speed
      }
      return this.bugTypeOptions[this.currentBugType].speed
    },
    preTunnelSpeedMultiplier() {
      const configured = parseFloat(this.bugsSettings.preTunnelSpeedMultiplier)
      return Number.isFinite(configured) && configured > 0 ? configured : 1
    },
    numFramesToRetreat() {
      return (this.bugsSettings.trialDuration || 1) * 60
    }
  },
  methods: {
    initiateStartPosition() {
      if (!this.exitHolePos || !this.entranceHolePos) {
        // Fallback to original logic if no holes data
        this.x = 0
        this.y = this.initialYs[randomRange(0, this.initialYs.length)]
        return
      }
      
      // Start from entrance hole center, like holesBug
      this.x = this.entranceHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      this.y = this.entranceHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      
      // Set target as exit hole center
      this.xTarget = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      this.yTarget = this.exitHolePos[1] + (this.bugsSettings.holeSize[1] / 2)
      
      // Determine movement direction like holesBug
      // If coming from left hole, move counterclockwise; if from right hole, move clockwise
      this.isCounterClockWise = this.entranceHolePos[0] < this.exitHolePos[0] // left to right entrance
      
      // Calculate and store original distance between holes for radius calculation
      this.originalDistanceBetweenHoles = Math.abs(this.xTarget - this.x)
      const oneThirdDistance = this.originalDistanceBetweenHoles / 3
      
      // Set horizontal target at 1/3 distance from current position towards target
      if (this.isRightExit) {
        this.horizontalTargetX = this.x + oneThirdDistance
      } else {
        this.horizontalTargetX = this.x - oneThirdDistance
      }
      
      // Set up initial horizontal movement
      this.setupHorizontalMovement()
      
      console.log(`Bug starting from entrance hole at`, {x: this.x, y: this.y}, 'target 1/3 distance:', this.horizontalTargetX)
    },
    
    setupHorizontalMovement() {
      // Set initial direction for horizontal movement like "low_horizontal" in holesBug
      this.directionAngle = this.isRightExit ? 2 * Math.PI : Math.PI
      const horizontalSpeed = this.currentSpeed * this.preTunnelSpeedMultiplier
      this.vx = horizontalSpeed * Math.cos(this.directionAngle)
      this.vy = horizontalSpeed * Math.sin(this.directionAngle)
      
      this.isMovingHorizontally = true
      this.isCircleTrackReached = false
      this.circleTransitionStarted = false
      this.frameCounter = 0
      
      console.log(`Horizontal movement setup, target: ${this.horizontalTargetX}`)
    },
    
    setupCircleMovement() {
      // Set circle center to the center of the tunnel image
      if (this.tunnelImagePos) {
        this.r0 = [
          this.tunnelImagePos.x + (this.tunnelImagePos.width / 2),
          this.tunnelImagePos.y + (this.tunnelImagePos.height / 2)
        ]
      } else {
        // Fallback to screen center
        this.r0 = [this.canvas.width / 2, this.canvas.height / 2]
      }
      
      // Set radius using original distance like holesBug implementation
      this.r = this.originalDistanceBetweenHoles * this.bugsSettings.circleRadiusScale
      
      // Calculate the bottom point of the circle for smooth entry
      // The bug should transition to the bottom of the circle, not jump to a random point
      this.theta = Math.PI / 2 // Bottom of circle (90 degrees)
      
      this.isMovingHorizontally = false
      this.circleTransitionStarted = true
      this.isCircleTrackReached = false
      this.isHoleRetreatStarted = false
      
      // Don't start custom exit timer - let bugsMixin handle retreat logic
    },
    
    move() {
      if (this.isDead || this.isRetreated) {
        this.draw()
        return
      }
      
      this.frameCounter++
      
      // Use existing bugsMixin logic to determine when to retreat
      // This will handle app signals and experiment timers
      if (this.isCircleTrackReached && !this.isHoleRetreatStarted) {
        // Use the same retreat logic as holesBug
        this.checkHoleRetreat()
      }
      
      if (this.isHoleRetreatStarted) {
        this.moveToExitHole()
      } else if (this.isMovingHorizontally) {
        this.horizontalMove()
      } else if (this.circleTransitionStarted && !this.isCircleTrackReached) {
        this.transitionToCircle()
      } else if (this.isCircleTrackReached) {
        this.circularMove()
      }
      
      this.draw()
    },
    
    horizontalMove() {
      // Move horizontally like "low_horizontal" movement in holesBug
      this.x += this.vx
      this.y += this.vy
      
      // Check if we've reached 1/3 of screen width
      let reachedTarget = false
      if (this.isRightExit && this.x >= this.horizontalTargetX) {
        reachedTarget = true
      } else if (!this.isRightExit && this.x <= this.horizontalTargetX) {
        reachedTarget = true
      }
      
      if (reachedTarget) {
        console.log('Reached 1/3 screen, starting circle transition')
        this.circleTransitionStarted = true
        this.setupCircleMovement()
      }
    },
    
    transitionToCircle() {
      // Smooth transition from horizontal movement to the bottom of the circle
      const targetX = this.r0[0] + this.r * Math.cos(this.theta)
      const targetY = this.r0[1] + this.r * Math.sin(this.theta)
      
      const dx = targetX - this.x
      const dy = targetY - this.y
      const distance = Math.sqrt(dx * dx + dy * dy)
      
      if (distance < 20) {
        // Close enough to circle, start circular movement
        this.isCircleTrackReached = true
        this.x = targetX
        this.y = targetY
        console.log('Circle movement started at bottom of circle')
      } else {
        // Move towards the bottom point of the circle smoothly
        const transitionSpeed = (this.currentSpeed || 5) * this.preTunnelSpeedMultiplier
        this.x += (dx / distance) * transitionSpeed * 0.98 // Slower transition
        this.y += (dy / distance) * transitionSpeed * 0.98
      }
    },
    
    circularMove() {
      // Circular movement exactly like in holesBug
      const speed = this.vTheta || this.currentSpeed || 5
      this.theta += Math.abs(speed) * Math.sqrt(2) / this.r
      
      // Keep theta in range
      if (this.theta > 2 * Math.PI) {
        this.theta -= 2 * Math.PI
      }
      
      this.x = this.r0[0] + (this.r * Math.cos(this.theta)) * (this.isCounterClockWise ? -1 : 1)
      this.y = this.r0[1] + this.r * Math.sin(this.theta)
    },
    
    // Use holesBug retreat logic
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
    
    moveToExitHole() {
      // Linear movement from current position to exit hole center using retreat speeds
      this.x += this.vx
      this.y += this.vy
      
      // Check if reached exit hole
      if (this.isInsideExitHoleBoundaries()) {
        this.hideBug()
      }
    },
    
    isInsideExitHoleBoundaries() {
      // Check if bug is inside exit hole boundaries
      const holeSize = this.bugsSettings.holeSize
      const exitX = this.exitHolePos[0]
      const exitY = this.exitHolePos[1]
      
      return this.x >= exitX && this.x <= exitX + holeSize[0] &&
             this.y >= exitY && this.y <= exitY + holeSize[1]
    },
    
    straightMove(noiseWeight = null) {
      // Keep original straight move for fallback
      let xNoise = this.y > this.canvas.height / 2 ? 0 : 0.2 * this.getRandomNoise()
      let speedWeight = this.y < this.canvas.height / 2 ? 0.4 : 1
      this.dx = (this.vx * speedWeight) + xNoise
      this.dy = (this.vy * speedWeight)
      this.x += this.dx
      this.y += this.dy
    },
    
    getAngleRadians() {
      // Use the same rotation logic as holesBug for circle movement
      if (this.isMoveInCircles && this.isCircleTrackReached && !this.isHoleRetreatStarted) {
        return Math.atan2(this.y - this.r0[1], this.x - this.r0[0]) + (this.isCounterClockWise ? 0 : Math.PI)
      }
      return Math.atan2(this.vy, this.vx) + Math.PI / 2
    }
  }
}
</script>

<style scoped>

</style>
