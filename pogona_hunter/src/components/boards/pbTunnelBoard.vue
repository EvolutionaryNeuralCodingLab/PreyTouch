<template>
  <div class="board-canvas-wrapper" oncontextmenu="return false;" v-on:mousedown="analyzeScreenTouch">
  <div id="bugs-board">
    <audio ref="audio1">
      <source src="../../assets/sounds/2.mp3" type="audio/mpeg">
    </audio>
    <canvas id="backgroundCanvas" v-bind:style="{background: bugsSettings.backgroundColor}"
            v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"></canvas>
    <canvas id="tunnelCanvas" v-bind:height="tunnelHeight" v-bind:width="tunnelWidth"
            v-on:mousedown="handleTunnelClick($event)"></canvas>
    <canvas id="bugCanvas" v-bind:height="canvasParams.height" v-bind:width="canvasParams.width"
            v-on:mousedown="setCanvasClick($event)">
      <pb-tunnel-bug v-for="(value, index) in bugsProps"
                 :key="index"
                 :bugsSettings="bugsSettings"
                 :tunnelImagePos="tunnelImagePos"
                 :exitHolePos="exitHolePos"
                 :entranceHolePos="entranceHolePos"
                 ref="bugChild"
                 v-on:bugRetreated="endTrial">
      </pb-tunnel-bug>
    </canvas>
  </div>
</div>
</template>

<script>
import boardsMixin from './boardsMixin'
import pbTunnelBug from '../bugs/pbTunnelBug.vue'
import defaultTunnel from '@/assets/curtains/Parsley.png'

const parseConfigNumber = (value, fallback) => {
  const parsed = parseFloat(value)
  return Number.isNaN(parsed) ? fallback : parsed
}
const DEFAULT_TUNNEL_IMAGE = process.env.TUNNEL_IMAGE || '@/assets/curtains/Parsley.png'
const tunnelAssetContext = require.context('@/assets', true, /\.(png|jpe?g|gif|svg|webp)$/)
const TUNNEL_IMAGE_Y_OFFSET = 10
const DEFAULT_TUNNEL_CUTOUTS = Object.freeze({
  enabled: true,
  squareSize: 150,
  radiusScale: 0.33,
  centerYOffset: 0,
  angles: [0, 120, 240]
})

const isAbsoluteSrc = (value) => /^(?:https?:)?\/\//i.test(value) || value.startsWith('/') || value.startsWith('data:')

const normalizeAssetPath = (value) => {
  if (!value) {
    return null
  }
  const trimmed = value.trim().replace(/^['"]|['"]$/g, '')
  if (trimmed.startsWith('@/assets/')) {
    return `.${trimmed.replace('@/assets', '')}`
  }
  if (trimmed.startsWith('./') || trimmed.startsWith('../')) {
    return trimmed
  }
  return `./${trimmed}`
}

const resolveTunnelAsset = (value) => {
  if (!value) {
    return null
  }
  if (isAbsoluteSrc(value)) {
    return value
  }
  const normalized = normalizeAssetPath(value)
  try {
    if (normalized && tunnelAssetContext.keys().includes(normalized)) {
      return tunnelAssetContext(normalized)
    }
  } catch (err) {
    console.warn(`[pbTunnelBoard] Unable to resolve tunnel  asset "${value}": ${err.message}`)
  }
  return null
}

export default {
  name: 'pbTunnelBoard',
  components: {pbTunnelBug},
  mixins: [boardsMixin],
  data() {
    return {
      bugsSettings: { // extends the mixin's bugSettings
        holeSize: [200, 200], // Same size as holesBoard
        exitHole: 'right', // Default exit hole
        entranceHole: null,
        holesHeightScale: 0.1, // Same as holesBoard
        circleHeightScale: 0.5,
        circleRadiusScale: 0.25,
        preTunnelSpeedMultiplier: 1.5,
        tunnelImage: process.env.TUNNEL_IMAGE || DEFAULT_TUNNEL_IMAGE,
        tunnelRotation: parseConfigNumber(process.env.TUNNEL_ROTATION, 60),
        tunnelScale: parseConfigNumber(process.env.TUNNEL_SCALE, 0.75),
        tunnelOpacity: parseConfigNumber(process.env.TUNNEL_OPACITY, 1)
      },
      tunnelImagePos: {
        x: 0,
        y: 0,
        width: 1500, // Increased to accommodate larger scaled image
        height: 1500 // Increased to accommodate larger scaled image
      },
      holes: {
        left: { x: 50, y: 0 },
        right: { x: 0, y: 0 }
      },
      imageLoaded: false,
      isSplitAnimating: false,
      splitAnimationFrame: null,
      xpad: 100, // padding for holes
      imageVisible: true, // Track if image is visible
      tunnelTouchPadding: 100, // Positive values expand the clickable region around the tunnel
      tunnelCutouts: {...DEFAULT_TUNNEL_CUTOUTS},
      tunnelTransparencyLog: []
    }
  },
  computed: {
    isSplitBugsView: function () {
      return this.bugsSettings.isSplitBugsView && this.bugsSettings.numOfBugs > 1
    },
    tunnelHeight() {
      return 1500 // Increased to accommodate larger scaled image
    },
    tunnelWidth() {
      return 1500 // Increased to accommodate larger scaled image
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
    exitHolePos: function () {
      // Use the same logic as holesBoard
      const exitHole = this.bugsSettings.exitHole
      return this.holesPositions[exitHole]
    },
    entranceHolePos: function () {
      // Use the same logic as holesBoard - entrance is opposite of exit
      let entranceHole = this.bugsSettings.exitHole === 'left' ? 'right' : 'left'
      return this.holesPositions[entranceHole]
    }
  },
  methods: {
    resetImageForNewTrial() {
      // Reset image visibility and canvas display for each new trial/session
      this.imageVisible = true
      this.imageLoaded = false
      this.tunnelTransparencyLog = []
      let canvas = document.getElementById('tunnelCanvas')
      if (canvas) {
        canvas.style.display = 'block'
        canvas.style.pointerEvents = 'auto'
        // Clear any previous image
        let ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      }
      console.log('Parsley image reset for new trial')
    },

    initDrawing() {
      // First reset the image for new trial
      this.resetImageForNewTrial()

      // then draw the holes
      this.drawHoles()
      // // Draw the background first
      // if (this.isSplitBugsView && this.bugsSettings.bugMappedBackground) {
      //   this.drawSplitBackground()
      // } else {
      //   this.drawSolidBackground()
      // }

      // // Flash photodiode square at trial start
      // this.drawSquareForPhotoDiode()

      // // Then draw the tunnel
      let canvas = document.getElementById('tunnelCanvas')
      // Position tunnel image in center of screen
      this.tunnelImagePos.x = (this.canvas.width / 2) - (this.tunnelWidth / 2)
      this.tunnelImagePos.y = (this.canvas.height / 2) - (this.tunnelHeight / 2)

      canvas.style.left = `${this.tunnelImagePos.x}px`
      canvas.style.top = `${this.tunnelImagePos.y}px`

      // Set up holes positions relative to the tunnel image (these are for bug spawning)
      this.holes.left.x = this.tunnelImagePos.x + 50
      this.holes.left.y = this.tunnelImagePos.y + (this.tunnelHeight / 2)
      this.holes.right.x = this.tunnelImagePos.x + this.tunnelWidth - 50
      this.holes.right.y = this.tunnelImagePos.y + (this.tunnelHeight / 2)

      let ctx = canvas.getContext('2d')
      // Don't set background color - keep transparent to utilize image transparency

      // ctx.fillStyle = this.bugsSettings.backgroundColor
      // ctx.fillRect(0, 0, canvas.width, canvas.height)
      const img = new Image()
      img.onload = () => {
        console.log('drawing image')
        // Draw image large but fit within canvas bounds
        ctx.save()
        ctx.translate(canvas.width / 2, canvas.height / 2) // Move to center

        // Scale the image to fit within the canvas while maintaining aspect ratio
        const baseScale = Math.min(canvas.width / img.width, canvas.height / img.height)
        const configuredScale = parseFloat(this.bugsSettings.tunnelScale)
        const scaleMultiplier = Number.isFinite(configuredScale) && configuredScale > 0 ? configuredScale : 0.75
        const finalScale = baseScale * scaleMultiplier
        ctx.scale(finalScale, finalScale)
        // Rotate the tunnel reward image using configured angle (degrees)
        const configuredRotation = parseFloat(this.bugsSettings.tunnelRotation)
        const rotationRad = Number.isFinite(configuredRotation) ? configuredRotation * (Math.PI / 180) : Math.PI / 3
        ctx.save()
        ctx.rotate(rotationRad)

        const configuredOpacity = parseFloat(this.bugsSettings.tunnelOpacity)
        const imageOpacity = Number.isFinite(configuredOpacity) ? Math.min(Math.max(configuredOpacity, 0), 1) : 1
        ctx.globalAlpha = imageOpacity

        ctx.drawImage(img, -img.width / 2, -img.height / 2 + TUNNEL_IMAGE_Y_OFFSET) // Use natural image size
        ctx.restore() // remove rotation so cutouts stay aligned to canvas axes
        const circleRadius = this.computeBugCircleRadius()
        this.applyTunnelCutouts(ctx, img.width, img.height, finalScale, circleRadius)
        ctx.restore()
        this.imageLoaded = true
      }
      img.src = this.getTunnelImageSource()
    },
    drawHoles() {
      let holeImage = new Image()
      let canvas = document.getElementById('backgroundCanvas')
      let ctx = canvas.getContext('2d')
      let [holeW, holeH] = this.bugsSettings.holeSize
      holeImage.src = require('@/assets/hole2.png')
      holeImage.onload = () => {
        Object.values(this.holesPositions).forEach(pos => {
          ctx.drawImage(holeImage, pos[0], pos[1], holeW, holeH)
          console.log('Drawing holes', pos[0], pos[1], holeW, holeH)
        })
      }
    },
    // Override setCanvasClick to gate touches behind the tunnel reveal
    setCanvasClick(event) {
      if (!this.$refs.bugChild || this.$refs.bugChild.length === 0) {
        return
      }

      const relativeX = event.x - this.canvas.offsetLeft
      const relativeY = event.y - this.canvas.offsetTop

      // Block bug interactions until the tunnel image has been removed
      if (this.imageLoaded && this.imageVisible) {
        if (this.checkImageClick(relativeX, relativeY)) {
          return
        }
        return
      }

      // Once the tunnel is gone, fall back to the shared touch pipeline so hits are logged normally
      this.handleTouchEvent(event.x, event.y)
    },
    checkImageClick(x, y) {
      // Check if click is within the tunnel image bounds
      const localX = x - this.tunnelImagePos.x
      const localY = y - this.tunnelImagePos.y
      const padding = Number(this.tunnelTouchPadding) || 0
      if (localX < -padding || localY < -padding ||
          localX > (this.tunnelWidth + padding) || localY > (this.tunnelHeight + padding)) {
        return false
      }
      console.log('Persil image clicked, splitting image')
      this.splitImageAndHide()
      return true
    },
    handleTunnelClick(event) {
      // This is for direct clicks on the tunnel canvas
      if (!this.imageLoaded || !this.imageVisible || !this.$refs.bugChild || this.$refs.bugChild.length === 0) return
      // Get click position relative to canvas
      let rect = event.target.getBoundingClientRect()
      let clickX = event.clientX - rect.left
      let clickY = event.clientY - rect.top
      const padding = Number(this.tunnelTouchPadding) || 0
      const withinBounds = (clickX >= -padding && clickX <= this.tunnelWidth + padding &&
        clickY >= -padding && clickY <= this.tunnelHeight + padding)
      if (withinBounds) {
        console.log('Persil image clicked directly, splitting image')
        this.splitImageAndHide()
      }
    },
    splitImageAndHide() {
      if (this.isSplitAnimating) {
        return
      }
      const canvas = document.getElementById('tunnelCanvas')
      if (!canvas) {
        this.hideImage()
        return
      }
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        this.hideImage()
        return
      }
      // Let touches pass through while the animation plays
      canvas.style.pointerEvents = 'none'
      // Allow bug hits to be registered while the split animation runs
      this.imageVisible = false
      let snapshot
      try {
        snapshot = new Image()
        snapshot.onload = () => this.animateSplit(snapshot, canvas, ctx)
        snapshot.src = canvas.toDataURL()
      } catch (err) {
        console.warn('Unable to create split animation', err)
        this.hideImage()
      }
    },
    animateSplit(snapshot, canvas, ctx) {
      const halfWidth = canvas.width / 2
      const duration = 600
      const maxOffsetX = this.tunnelWidth * 0.15
      const maxOffsetY = this.tunnelHeight * 0.05
      this.isSplitAnimating = true
      const startTime = performance.now()
      const step = (timestamp) => {
        const rawProgress = (timestamp - startTime) / duration
        const progress = Math.min(Math.max(rawProgress, 0), 1)
        const eased = 1 - Math.pow(1 - progress, 3)
        const offsetX = maxOffsetX * eased
        const offsetY = maxOffsetY * eased
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        ctx.drawImage(snapshot, 0, 0, halfWidth, canvas.height, -offsetX, -offsetY, halfWidth, canvas.height)
        ctx.drawImage(snapshot, halfWidth, 0, halfWidth, canvas.height, halfWidth + offsetX, offsetY, halfWidth, canvas.height)
        if (progress < 1) {
          this.splitAnimationFrame = requestAnimationFrame(step)
        } else {
          this.isSplitAnimating = false
          this.splitAnimationFrame = null
          this.hideImage()
        }
      }
      this.splitAnimationFrame = requestAnimationFrame(step)
    },

    isOpaqueTunnelPixel(x, y) {
      let canvas = document.getElementById('tunnelCanvas')
      if (!canvas) {
        return false
      }

      let ctx = canvas.getContext('2d')
      if (!ctx) {
        return false
      }

      try {
        let pixel = ctx.getImageData(Math.floor(x), Math.floor(y), 1, 1).data
        return pixel[3] > 0 // alpha channel
      } catch (err) {
        console.warn('Unable to read tunnel canvas pixel', err)
        return false
      }
    },
    hideImage() {
      // Simply hide the image by setting visibility
      if (this.splitAnimationFrame) {
        cancelAnimationFrame(this.splitAnimationFrame)
        this.splitAnimationFrame = null
      }
      this.isSplitAnimating = false
      this.imageVisible = false
      let canvas = document.getElementById('tunnelCanvas')
      if (canvas) {
        canvas.style.display = 'none'
        canvas.style.pointerEvents = 'none'
      }
      console.log('Persil image hidden')
    },

    getTunnelImageSource() {
      const configuredImage = this.bugsSettings.tunnelImage ||
        process.env.TUNNEL_IMAGE ||
        process.env.TUNNEL_FOOD_IMAGE ||
        null
      const resolvedAsset = resolveTunnelAsset(configuredImage)
      if (resolvedAsset) {
        return resolvedAsset
      }
      if (configuredImage && isAbsoluteSrc(configuredImage)) {
        return configuredImage
      }
      return defaultTunnel
    },

    onBoardAnimationFrame() {
      if (!this.imageLoaded || !this.imageVisible || !this.$refs.bugChild || this.$refs.bugChild.length === 0) {
        return
      }
      const timestamp = Date.now()
      const frameTime = typeof performance !== 'undefined' && typeof performance.now === 'function'
        ? performance.now()
        : timestamp
      this.$refs.bugChild.forEach((bug, index) => {
        if (!bug || !this.isPointUnderCutout(bug.x, bug.y)) {
          return
        }
        this.logTunnelTransparencySample(bug, index, timestamp, frameTime)
      })
    },

    logTunnelTransparencySample(bug, index, timestamp, frameTime) {
      const entry = {
        time: timestamp,
        frameTime,
        bugIndex: index,
        bugType: bug.currentBugType || null,
        x: bug.x,
        y: bug.y
      }
      if (bug.bugId) {
        entry.bugId = bug.bugId
      }
      this.tunnelTransparencyLog.push(entry)
    },

    isPointUnderCutout(screenX, screenY) {
      if (!this.imageLoaded || !this.imageVisible) {
        return false
      }
      const localX = screenX - this.tunnelImagePos.x
      const localY = screenY - this.tunnelImagePos.y
      if (localX < 0 || localX >= this.tunnelWidth || localY < 0 || localY >= this.tunnelHeight) {
        return false
      }
      return !this.isOpaqueTunnelPixel(localX, localY)
    },

    collectBoardMetrics() {
      if (!this.tunnelTransparencyLog || this.tunnelTransparencyLog.length === 0) {
        return null
      }
      const entries = this.tunnelTransparencyLog.slice()
      this.tunnelTransparencyLog = []
      return {tunnel_transparency: entries}
    },

    resolveTunnelCutoutConfig() {
      const base = {...DEFAULT_TUNNEL_CUTOUTS, ...(this.tunnelCutouts || {})}
      const overrides = this.bugsSettings.tunnelCutouts || {}
      const resolved = {...base, ...overrides}
      const baseAngles = Array.isArray(base.angles) ? base.angles : DEFAULT_TUNNEL_CUTOUTS.angles
      if (overrides.angles) {
        resolved.angles = Array.isArray(overrides.angles) ? [...overrides.angles] : [...baseAngles]
      } else {
        resolved.angles = [...baseAngles]
      }
      return resolved
    },

    computeBugCircleRadius() {
      if (!this.entranceHolePos || !this.exitHolePos) {
        return null
      }
      const entranceCenterX = this.entranceHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      const exitCenterX = this.exitHolePos[0] + (this.bugsSettings.holeSize[0] / 2)
      const distance = Math.abs(exitCenterX - entranceCenterX)
      const configuredScale = parseFloat(this.bugsSettings.circleRadiusScale)
      const scale = Number.isFinite(configuredScale) && configuredScale > 0
        ? configuredScale
        : DEFAULT_TUNNEL_CUTOUTS.radiusScale
      return distance * scale
    },

    applyTunnelCutouts(ctx, imageWidth, imageHeight, scaleFactor, circleRadiusPixels) {
      const config = this.resolveTunnelCutoutConfig()
      if (!config || !config.enabled) {
        return
      }
      const angles = Array.isArray(config.angles) ? config.angles : []
      if (angles.length === 0) {
        return
      }
      const parsedSize = parseFloat(config.squareSize)
      let squareSize = Number.isFinite(parsedSize) && parsedSize > 0 ? parsedSize : DEFAULT_TUNNEL_CUTOUTS.squareSize
      const scale = Number.isFinite(scaleFactor) && scaleFactor > 0 ? scaleFactor : 1
      const size = squareSize / scale
      const rectWidth = size * 0.25
      const rectHeight = size * 1.2
      const parsedRadiusScale = parseFloat(config.radiusScale)
      const radiusScale = Number.isFinite(parsedRadiusScale) && parsedRadiusScale > 0
        ? parsedRadiusScale
        : DEFAULT_TUNNEL_CUTOUTS.radiusScale
      const parsedRadius = parseFloat(config.radius)
      let desiredRadius = Number.isFinite(circleRadiusPixels) && circleRadiusPixels > 0
        ? circleRadiusPixels
        : null
      if (!desiredRadius) {
        desiredRadius = Number.isFinite(parsedRadius) && parsedRadius > 0
          ? parsedRadius
          : Math.min(imageWidth, imageHeight) * radiusScale
      }
      const radius = desiredRadius / scale
      const parsedCenterYOffset = parseFloat(config.centerYOffset)
      const centerYOffset = Number.isFinite(parsedCenterYOffset)
        ? parsedCenterYOffset
        : DEFAULT_TUNNEL_CUTOUTS.centerYOffset
      const centerX = 0
      const centerY = centerYOffset

      ctx.save()
      ctx.globalCompositeOperation = 'destination-out'
      ctx.globalAlpha = 1
      angles.forEach(angleValue => {
        const angle = parseFloat(angleValue)
        if (!Number.isFinite(angle)) {
          return
        }
        const radians = angle * (Math.PI / 180)
        const offsetX = radius * Math.sin(radians)
        const offsetY = -radius * Math.cos(radians)
        const centerPosX = centerX + offsetX
        const centerPosY = centerY + offsetY
        const rotation = this.resolveCutoutRotation(angle)
        ctx.save()
        ctx.translate(centerPosX, centerPosY)
        ctx.rotate(rotation)
        ctx.fillRect(-rectWidth / 2, -rectHeight / 2, rectWidth, rectHeight)
        ctx.restore()
      })
      ctx.restore()
    },
    resolveCutoutRotation(angleDegrees) {
      const normalized = ((angleDegrees % 360) + 360) % 360
      if (normalized === 0) {
        return Math.PI / 2
      }
      if (normalized === 240) {
        return (3 * Math.PI) / 4
      }
      return Math.PI / 4
    }
  }
}
</script>

<style scoped>

#bugCanvas {
  padding: 0;
  z-index: 3; /* Behind the tunnel image */
  display: block;
  position: absolute;
  bottom: 0;
  top: auto;
}

#tunnelCanvas {
  padding: 0;
  z-index: 4; /* On top of bugs to utilize transparency */
  display: block;
  position: absolute;
  top: auto;
  bottom: 0;
}

</style>
