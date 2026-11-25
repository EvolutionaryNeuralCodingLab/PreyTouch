<template>
  <span hidden></span>
</template>
<script>
import {distance, randomRange} from '../../js/helpers'
const config = require('../../config.json')
const defaultBugType = (process.env.DEFAULT_CUSTOM_BUG_TYPE || 'worm').trim() || 'worm'

export default {
  name: 'customBug',

  props: {
    bugId: Number,
    bugsSettings: Object,
    trajectory: {
      type: Object,
      required: true
    }
  },

  data() {
    return {
      // bugTypeOptions: config.bugTypes,
      // rowIndex: 0,
      bugTypeOptions: config.bugTypes,
      trajectoryFit: {scale: 1, offsetX: 0, offsetY: 0},
      rowIndex: 0,
      imageStep: 0,
      currentBugSize: 0,
      frameInterval: 0,
      lastUpdateTime: null,
      isPrepared: false,
      isPlaying: false,
      isDead: false,
      isRetreated: false
    }
  },

  created() {
    this.imagesBySource = Object.create(null) // Image buffers do not need Vue observation.
    this.defaultSizesBySource = Object.create(null)
    this.loggedSourcesBySource = Object.create(null)
    this.imageLoadState = null
    this.ctx = null
  },

  computed: {

    currentRow() {
      return this.trajectory.rows[this.rowIndex]
    },

    currentBugType() {
      return this.getRowImageSource(this.currentRow)
    },

    // x() {
    //   return this.currentRow.x
    // },

    // y() {
    //   return this.currentRow.y
    // },

    x() {
      return this.trajectoryFit.offsetX + this.currentRow.x * this.trajectoryFit.scale
    },

    y() {
      return this.trajectoryFit.offsetY + this.currentRow.y * this.trajectoryFit.scale
    },

    isStrikeable() {
      return this.csvFlag(this.currentRow['strike-ability'])
    },

    isRewardBug() {
      return this.csvFlag(this.currentRow['is-reward'])
    }
  },

  beforeDestroy() {
    this.stopPlayback()
  },

  methods: {

    csvFlag(value) {
      if (typeof value === 'string') {
        value = value.trim().toLowerCase()
      }
      return value === true || value === 1 || value === '1' || value === 'true' // Keep "0" and "false" false.
    },

    validateTrajectory() {
      if (
        !this.trajectory ||
        !Array.isArray(this.trajectory.rows) ||
        this.trajectory.rows.length === 0
      ) {
        throw new Error('Custom trajectory must contain at least one row')
      }
      this.trajectory.rows.forEach((row, index) => {
        const label = 'Custom trajectory row ' + (index + 1) + ': '
        if (!row || typeof row !== 'object' || Array.isArray(row)) {
          throw new Error(label + 'must be an object')
        }
        ;['x', 'y', 'angle'].forEach(field => {
          if (!Number.isFinite(row[field])) {
            throw new Error(label + field + ' must be a finite number')
          }
        })
        if (row.size !== null && (!Number.isFinite(row.size) || row.size < 0)) {
          throw new Error(label + 'size must be null or a nonnegative number')
        }
        if (row['img-src'] != null && typeof row['img-src'] !== 'string') {
          throw new Error(label + 'img-src must be a string or null')
        }
        ;['strike-ability', 'is-reward'].forEach(field => {
          let value = row[field]
          if (typeof value === 'string') {
            value = value.trim().toLowerCase()
          }
          if (value != null && ![true, false, 0, 1, '0', '1', 'true', 'false'].includes(value)) {
            throw new Error(label + field + ' must be a boolean or 0/1')
          }
        })
      })
    },

    async prepare(ctx, pool = null) {
      this.stopPlayback() // A new preparation invalidates any unfinished image loads.
      this.validateTrajectory()
      // this.ctx = ctx
      this.ctx = ctx

      // Custom CSV coordinates are authored on a 1920 × 1080 canvas.
      const referenceWidth = 1920
      const referenceHeight = 1080
      const {width, height} = ctx.canvas
      const scale = Math.min(width / referenceWidth, height / referenceHeight)

      this.trajectoryFit = {
        scale,
        offsetX: (width - referenceWidth * scale) / 2,
        offsetY: (height - referenceHeight * scale) / 2
      }
      this.isDead = false
      this.isRetreated = false
      this.rowIndex = 0
      this.imageStep = 0
      this.currentBugSize = 0
      this.lastUpdateTime = null
      const load = {
        cancelled: false,
        pending: new Set(),
        imagesBySource: Object.create(null),
        loggedSourcesBySource: Object.create(null)
      }
      load.imageRequests = pool ? pool.requests : new Map()
      load.imagePending = pool ? pool.pending : load.pending
      this.imageLoadState = load
      try {
        if (this.bugsSettings.bugFps === 'native') {
          this.frameInterval = 0
        } else {
          const fps = Number(this.bugsSettings.bugFps)
          if (!Number.isFinite(fps) || fps <= 0) {
            throw new Error('bugFps must be "native" or a positive number')
          }
          this.frameInterval = 1000 / fps
        }
        await this.loadTrajectoryImages(load)
        if (!this.isCurrentImageLoad(load)) {
          return
        }
        const size = this.getDrawSize(this.getCurrentImage())
        this.currentBugSize = Math.max(size.width, size.height)
        this.isPrepared = true
      } catch (error) {
        if (!this.isCurrentImageLoad(load)) {
          return
        }
        this.stopPlayback()
        throw error
      }
    },

    startPlayback() {
      if (!this.isPrepared || this.isRetreated) {
        return
      }
      this.lastUpdateTime = null
      this.isPlaying = true
    },

    stopPlayback() {
      this.isPlaying = false
      this.isPrepared = false
      this.isRetreated = true
      const load = this.imageLoadState
      this.imageLoadState = null
      if (load) {
        load.cancelled = true
        Array.from(load.pending).forEach(cancel => cancel())
        load.imagesBySource = Object.create(null)
        load.loggedSourcesBySource = Object.create(null)
      }
      this.imagesBySource = Object.create(null)
      this.defaultSizesBySource = Object.create(null)
      this.loggedSourcesBySource = Object.create(null)
      this.ctx = null
    },

    startRetreat() {
      if (this.isRetreated) {
        return
      }
      this.stopPlayback()
      this.$emit('bugRetreated')
    },

    getRowImageSource(row) {
      return typeof row['img-src'] === 'string' && row['img-src'].trim()
        ? row['img-src'].trim()
        : defaultBugType
    },

    getAngleRadians() {
      return this.currentRow.angle
    },

    getImageSources(source) {
      if (Object.keys(this.bugTypeOptions).includes(source)) {
        const count = this.bugTypeOptions[source].numImagesPerBug
        return Array.from({length: count}, (_, index) => {
          return require('@/assets/' + source + index + '.png')
        })
      }
      const filename = source.startsWith('/') ? source : '/' + source
      try {
        return [require('@/assets' + filename)]
      } catch (error) {
        return [source]
      }
    },

    getDeadImageSource(source) {
      if (!Object.keys(this.bugTypeOptions).includes(source)) {
        return null
      }
      return require('@/assets/' + source + '_dead.png')
    },

    isCurrentImageLoad(load) {
      return this.imageLoadState === load && !load.cancelled && !this.isRetreated
    },

    loadImage(source, load, label = source) {
      if (!this.isCurrentImageLoad(load)) {
        return Promise.resolve(null)
      }

      if (load.imageRequests.has(source)) {
        return load.imageRequests.get(source) // Share an image even while it is loading.
      }

      const imageLabel = /^data:/i.test(label) ? 'inline image' : label
      const request = new Promise((resolve, reject) => {
        let image = new Image()
        let settled = false

        const finish = (value, error) => {
          if (settled) return
          settled = true
          image.onload = null
          image.onerror = null
          load.imagePending.delete(cancel)
          image = null

          if (error) {
            reject(error)
          } else {
            resolve(value)
          }
        }

        const cancel = () => finish(null)
        load.imagePending.add(cancel)

        image.onload = () => {
          if (settled) return
          if (!(image.naturalWidth > 0 && image.naturalHeight > 0)) {
            finish(null, new Error('Bug image has no usable dimensions: ' + imageLabel))
            return
          }
          finish(image)
        }

        image.onerror = () => finish(null, new Error('Could not load bug image: ' + imageLabel))

        try {
          image.src = source
        } catch (error) {
          finish(null, new Error('Could not load bug image: ' + imageLabel))
        }
      })

      load.imageRequests.set(source, request)
      return request
    },

    async loadTrajectoryImages(load) {
      const sources = Array.from(new Set(
        this.trajectory.rows.map(row => this.getRowImageSource(row))
      ))
      sources.forEach((source, index) => {
        load.loggedSourcesBySource[source] = /^data:/i.test(source)
          ? 'inline-image-' + index
          : source
      })
      let nextSource = 0
      const loadNextSource = async () => {
        while (nextSource < sources.length && this.isCurrentImageLoad(load)) {
          const source = sources[nextSource++]
          const label = load.loggedSourcesBySource[source]
          const images = []
          const paths = this.getImageSources(source)
          for (const path of paths) {
            const image = await this.loadImage(path, load, label)
            if (!this.isCurrentImageLoad(load)) {
              return
            }
            images.push(image)
          }
          const deadSource = this.getDeadImageSource(source)
          const deadImage = deadSource ? await this.loadImage(deadSource, load, label + '_dead') : null
          if (!this.isCurrentImageLoad(load)) {
            return
          }
          load.imagesBySource[source] = {images, deadImage}
        }
      }
      await Promise.all(Array.from({length: Math.min(4, sources.length)}, loadNextSource)) // Bound concurrent image requests.
      if (this.isCurrentImageLoad(load)) {
        this.imagesBySource = load.imagesBySource
        this.loggedSourcesBySource = load.loggedSourcesBySource
      }
    },

    advanceRow() {
      if (this.isDead || this.isRetreated) {
        return
      }
      const previousSource = this.getRowImageSource(this.currentRow)
      this.rowIndex = (this.rowIndex + 1) % this.trajectory.rows.length
      const currentSource = this.getRowImageSource(this.currentRow)
      this.imageStep = currentSource === previousSource
        ? this.imageStep + 1
        : 0
    },

    getCurrentImage() {
      const imageSet = this.imagesBySource[this.currentBugType]
      if (!imageSet) {
        return null
      }
      if (this.isDead && imageSet.deadImage) {
        return imageSet.deadImage
      }
      return imageSet.images[this.getCurrentImageIndex()]
    },

    getCurrentImageIndex() {
      const imageSet = this.imagesBySource[this.currentBugType]
      if (!imageSet || (this.isDead && imageSet.deadImage)) {
        return null
      }
      const images = imageSet.images
      if (images.length === 1) {
        return 0
      }
      const stepsPerImage = this.bugTypeOptions[this.currentBugType].stepsPerImage
      return Math.floor(this.imageStep / stepsPerImage) % images.length
    },

    getLoggedBugType() {
      const source = this.currentBugType
      if (!/^data:/i.test(source)) {
        return source
      }
      if (this.loggedSourcesBySource[source]) {
        return this.loggedSourcesBySource[source]
      }
      const sources = Array.from(new Set(this.trajectory.rows.map(row => this.getRowImageSource(row))))
      return 'inline-image-' + sources.indexOf(source) // Keep logs compact even before prepare or after stop.
    },

    getLoggedImageSource() {
      const source = this.currentBugType
      if (!Object.keys(this.bugTypeOptions).includes(source)) {
        return this.getLoggedBugType()
      }
      const frame = this.getCurrentImageIndex()
      return 'assets/' + source + (frame === null ? '_dead' : frame) + '.png'
    },

    getDrawSize(image) {
      if (!image) {
        throw new Error('Custom bug image is not loaded')
      }
      let size = this.currentRow.size
      if (size === null) {
        const source = this.currentBugType
        if (this.defaultSizesBySource[source] === undefined) {
          const options = Object.keys(this.bugTypeOptions).includes(source)
            ? this.bugTypeOptions[source]
            : null
          this.defaultSizesBySource[source] = options
            ? randomRange(options.radiusRange.min, options.radiusRange.max)
            : 0
        }
        size = this.defaultSizesBySource[source]
      }
      return {
        width: size === 0 ? image.naturalWidth : size,
        height: size === 0 ? image.naturalHeight : size
      }
    },

    move(bugs, now = performance.now()) {
      if (!this.isPlaying || this.isRetreated) {
        return
      }
      if (!this.isDead) {
        if (this.lastUpdateTime === null) {
          this.lastUpdateTime = now
        } else if (this.frameInterval === 0) {
          this.advanceRow()
          this.lastUpdateTime = now
        } else {
          const elapsed = now - this.lastUpdateTime
          const tolerance = Number.EPSILON * Math.max(now, this.frameInterval)
          const intervals = Math.floor((elapsed + tolerance) / this.frameInterval)
          // if (intervals > 0) {
          //   this.advanceRow() // Advance at most one CSV row per animation frame.
          //   const remainder = Math.min(this.frameInterval, Math.max(0, elapsed - this.frameInterval))
          //   this.lastUpdateTime = now - remainder
          // }
          if (intervals > 0) {
            const rowCount = this.trajectory.rows.length
            let steps = intervals

            // After a long pause, skip full loops while preserving the sprite animation.
            if (steps >= 2 * rowCount) {
              steps = rowCount + (intervals % rowCount)
              this.imageStep += intervals - steps
            }

            for (let i = 0; i < steps; i++) {
              this.advanceRow()
            }

            const remainder = Math.max(0, elapsed - intervals * this.frameInterval)
            this.lastUpdateTime = now - remainder
          }
        }
      }
      this.draw()
    },

    draw() {
      const image = this.getCurrentImage()
      if (!image || this.isRetreated) {
        return
      }
      const {width, height} = this.getDrawSize(image)
      this.currentBugSize = Math.max(width, height)
      this.ctx.save()
      try {
        this.ctx.translate(this.x, this.y)
        this.ctx.rotate(this.getAngleRadians())
        this.ctx.drawImage(image, -width / 2, -height / 2, width, height)
      } finally {
        this.ctx.restore()
      }
      if (!this.bugsSettings.isLogTrajectory) return
      this.$emit('trajectorySample', {
        time: Date.now(),
        in_block_trial_id: this.bugsSettings.trialID,
        trial_id: this.bugsSettings.trialDBId,
        bug_id: this.bugId,
        trajectory_id: this.trajectory.id,
        trajectory_name: this.trajectory.name,
        row_index: this.rowIndex,
        x: this.x,
        y: this.y,
        angle: this.getAngleRadians(),
        size: this.currentBugSize,
        width,
        height,
        img_src: this.getLoggedBugType(),
        image_src: this.getLoggedImageSource(),
        frame_index: this.getCurrentImageIndex(),
        is_strikeable: this.isStrikeable,
        is_reward_bug: this.isRewardBug,
        is_dead: this.isDead
      })
    },

    hitDistance(x, y) {
      return distance(x, y, this.x, this.y)
    },

    isHit(x, y) {
      if (!this.isPlaying || this.isDead || this.isRetreated || !this.isStrikeable) {
        return false
      }
      return this.hitDistance(x, y) <= this.currentBugSize * this.bugsSettings.hitRadiusScale
    },

    jump() {},

    flipCircle() {},

    escape() {}
  }
}
</script>