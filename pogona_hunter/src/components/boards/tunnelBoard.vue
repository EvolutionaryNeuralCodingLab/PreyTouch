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
      <tunnel-bug v-for="(value, index) in bugsProps"
                 :key="index"
                 :bugsSettings="bugsSettings"
                 :tunnelImagePos="tunnelImagePos"
                 :exitHolePos="exitHolePos"
                 :entranceHolePos="entranceHolePos"
                 ref="bugChild"
                 v-on:bugRetreated="endTrial">
      </tunnel-bug>
    </canvas>
  </div>
</div>
</template>

<script>
import boardsMixin from './boardsMixin'
import tunnelBug from '../bugs/tunnelBug.vue'
import {getKeyWithMinFirstArrayValue} from '../../js/helpers'
import defaultTunnelFood from '@/assets/curtains/Parsley.png'

const tunnelFoodAssetContext = require.context('@/assets', true, /\.(png|jpe?g|gif|svg|webp)$/)

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

const resolveTunnelFoodAsset = (value) => {
  if (!value) {
    return null
  }
  if (isAbsoluteSrc(value)) {
    return value
  }
  const normalized = normalizeAssetPath(value)
  try {
    if (normalized && tunnelFoodAssetContext.keys().includes(normalized)) {
      return tunnelFoodAssetContext(normalized)
    }
  } catch (err) {
    console.warn(`[tunnelBoard] Unable to resolve tunnel food asset "${value}": ${err.message}`)
  }
  return null
}

export default {
  name: 'tunnelBoard',
  components: {tunnelBug},
  mixins: [boardsMixin],
  data() {
    return {
      bugsSettings: { // extends the mixin's bugSettings
        holeSize: [200, 200], // Same size as holesBoard
        exitHole: 'right', // Default exit hole
        entranceHole: null,
        holesHeightScale: 0.1, // Same as holesBoard
        circleHeightScale: 0.5,
        circleRadiusScale: 0.2,
        preTunnelSpeedMultiplier: 1.5,
        tunnelFoodOpacity: 1
      },
      tunnelImagePos: {
        x: 0,
        y: 0,
        width: 1200, // Increased to accommodate larger scaled image
        height: 1500 // Increased to accommodate larger scaled image
      },
      holes: {
        left: { x: 50, y: 0 },
        right: { x: 0, y: 0 }
      },
      imageLoaded: false,
      xpad: 100, // padding for holes
      imageVisible: true // Track if image is visible
    }
  },
  computed: {
    tunnelHeight() {
      return 1500 // Increased to accommodate larger scaled image
    },
    tunnelWidth() {
      return 1200 // Increased to accommodate larger scaled image
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
      let canvas = document.getElementById('tunnelCanvas')
      if (canvas) {
        canvas.style.display = 'block'
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
      const img = new Image()
      img.onload = () => {
        console.log('drawing persil image')
        // Draw image large but fit within canvas bounds
        ctx.save()
        ctx.translate(canvas.width / 2, canvas.height / 2) // Move to center

        // Scale the image to fit within the canvas while maintaining aspect ratio
        const baseScale = Math.min(canvas.width / img.width, canvas.height / img.height)
        const configuredScale = parseFloat(this.bugsSettings.tunnelFoodScale)
        const scaleMultiplier = Number.isFinite(configuredScale) && configuredScale > 0 ? configuredScale : 0.75
        const finalScale = baseScale * scaleMultiplier
        ctx.scale(finalScale, finalScale)

        // Rotate the tunnel reward image using configured angle (degrees)
        const configuredRotation = parseFloat(this.bugsSettings.tunnelFoodRotation)
        const rotationRad = Number.isFinite(configuredRotation) ? configuredRotation * (Math.PI / 180) : Math.PI / 3
        ctx.rotate(rotationRad)

        const configuredOpacity = parseFloat(this.bugsSettings.tunnelFoodOpacity)
        const imageOpacity = Number.isFinite(configuredOpacity) ? Math.min(Math.max(configuredOpacity, 0), 1) : 1
        ctx.globalAlpha = imageOpacity
        
        ctx.drawImage(img, -img.width / 2, -img.height / 2 + 10) // Use natural image size
        ctx.restore()
        this.imageLoaded = true
      }
      const configuredImage = this.bugsSettings.tunnelFoodImage || process.env.TUNNEL_FOOD_IMAGE || null
      const resolvedAsset = resolveTunnelFoodAsset(configuredImage) || defaultTunnelFood
      img.src = resolvedAsset
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
    
    // Override setCanvasClick to handle both bug hits and image clicks
    setCanvasClick(event) {
      if (this.isHandlingTouch || !this.$refs.bugChild || this.$refs.bugChild.length === 0) {
        return
      }
      
      const x = event.x - this.canvas.offsetLeft
      const y = event.y - this.canvas.offsetTop

      // Force the first touch to interact with the tunnel image before bugs can be hit
      if (this.imageLoaded && this.imageVisible) {
        this.checkImageClick(x, y)
        return
      }
      
      // First check for bug hits
      let strikeDistances = {}
      let isRewardAnyTouch = Math.random() < this.bugsSettings.rewardAnyTouchProb
      
      for (let i = 0; i < this.$refs.bugChild.length; i++) {
        let bug = this.$refs.bugChild[i]
        if (bug.isDead || bug.isRetreated) {
          continue
        }
        let isRewardBug = this.bugsSettings.rewardBugs.includes(bug.currentBugType)
        strikeDistances[i] = [bug.hitDistance(x, y), bug.isHit(x, y), isRewardBug]
      }
      
      if (Object.keys(strikeDistances).length > 0) {
        // Get the bug with the minimum distance from the touch point
        let i = Number(getKeyWithMinFirstArrayValue(strikeDistances))
        let bug = this.$refs.bugChild[i]
        let isHit = strikeDistances[i][1]
        let isRewardBug = strikeDistances[i][2]
        
        if (isHit || isRewardAnyTouch) {
          this.handleTouchEvent(event.x, event.y) // Use parent's touch handler
          return
        }
      }
    },
    
    checkImageClick(x, y) {
      // Check if click is within the tunnel image bounds
      const localX = x - this.tunnelImagePos.x
      const localY = y - this.tunnelImagePos.y
      if (localX < 0 || localY < 0 || localX >= this.tunnelWidth || localY >= this.tunnelHeight) {
        return false
      }

      if (!this.isOpaqueTunnelPixel(localX, localY)) {
        return false
      }

      console.log('Persil image clicked, hiding image')
      this.hideImage()
      return true
    },
    
    handleTunnelClick(event) {
      // This is for direct clicks on the tunnel canvas
      if (!this.imageLoaded || !this.imageVisible || !this.$refs.bugChild || this.$refs.bugChild.length === 0) return
      
      // Get click position relative to canvas
      let rect = event.target.getBoundingClientRect()
      let clickX = event.clientX - rect.left
      let clickY = event.clientY - rect.top
      
      // Check if click is within the image bounds
      if (clickX >= 0 && clickX <= this.tunnelWidth && 
          clickY >= 0 && clickY <= this.tunnelHeight &&
          this.isOpaqueTunnelPixel(clickX, clickY)) {
        console.log('Persil image clicked directly, hiding image')
        this.hideImage()
      }
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
      this.imageVisible = false
      let canvas = document.getElementById('tunnelCanvas')
      canvas.style.display = 'none'
      
      console.log('Persil image hidden')
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
