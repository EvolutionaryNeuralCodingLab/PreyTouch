<template>
  <div id="custom-board" @contextmenu.prevent>
    <canvas
      ref="backgroundCanvas"
      id="customBackgroundCanvas"
      :width="canvasParams.width"
      :height="canvasParams.height"
      :style="{background: bugsSettings.backgroundColor}">
    </canvas>
    <canvas
      ref="bugCanvas"
      id="customBugCanvas"
      :width="canvasParams.width"
      :height="canvasParams.height"
      @mousedown="setCanvasClick">
    </canvas>
    <custom-bug
      v-for="value in bugsProps"
      :key="value.slotKey"
      ref="bugChild"
      :bug-id="value.slotIndex"
      :bugs-settings="bugsSettings"
      :trajectory="value.trajectory"
      @trajectorySample="recordCustomTrajectory($event, value.slotKey)"
      @bugRetreated="endTrial" />
  </div>
</template>

<script>
import { v4 as uuidv4 } from 'uuid'
import {getKeyWithMinFirstArrayValue} from '../../js/helpers'
import customBug from '../bugs/customBug.vue'

function defaultSettings() {
  return {
    numOfBugs: 0,
    movementType: 'custom',
    trialID: null,
    trialDBId: null,
    trialDuration: 5,
    speed: 0,
    bugFps: 60,
    isLogTrajectory: false,
    bloodDuration: 2000,
    backgroundColor: '#e8eaf6',
    hitRadiusScale: 2 / 3,
    isKillingAllByOneHit: process.env.IS_KILLING_ALL_BY_ONE_HIT,
    holeSize: [200, 200],
    holesHeightScale: 0.1,
    exitHole: 'right'
  }
}

export default {
  name: 'customBoard',
  components: {customBug},

  data() {
    return {
      bugsProps: [],
      bugsSettings: defaultSettings(),
      canvasParams: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      isHandlingTouch: false,
      isClimbing: false,
      isBloodOnScreen: false,
      isRewardGiven: false
    }
  },

  created() {
    this.identifier = uuidv4()
    this.trialToken = 0
    this.isDisposed = false
    this.activeBugs = []
    this.trialData = {}
    this.bugTrajectoryLog = [] // Keep frame samples outside Vue's reactive data.
    this.bugTrajectoryAnimationLog = []
    this.eventsLog = []
    this.trajectoryBugCount = 0
    this.touchesCounter = 0
    this.animationHandler = null
    this.bloodTimeout = null
    this.touchTimeout = null
    this.climbingTimeout = null
    this.rewardTimeout = null
    this.squareTimeout = null
    this.holeImage = null
    this.holeImagePromise = null
    this.finishHoleImage = null
    this.socketOpenHandler = () => {
      console.log('Custom board WebSocket connected')
    }
    this.socketCloseHandler = () => {
      if (this.bugsProps.length > 0) {
        console.warn('Custom trial stopped after WebSocket disconnected; unsent trial metrics were not saved')
        this.clearBoard()
      }
    }
    this.$socketClient.onOpen = this.socketOpenHandler
    this.$socketClient.onClose = this.socketCloseHandler
    this.$socketClient.subscribe({
      'cmd/visual_app/init_bugs': options => this.initBoard(options),
      'cmd/visual_app/hide_bugs': () => this.endTrial(),
      'cmd/visual_app/reload_app': () => location.reload(),
      'cmd/visual_app/reward_given': () => {
        this.isRewardGiven = true
        clearTimeout(this.rewardTimeout)
        this.rewardTimeout = setTimeout(() => {
          this.rewardTimeout = null
          this.isRewardGiven = false
        }, 20 * 1000)
      },
      'cmd/visual_app/healthcheck': payload => {
          if (payload) return  // A probe has an empty payload. A reply contains JSON.

          this.publish('cmd/visual_app/healthcheck', JSON.stringify({
              id: this.identifier,
              host: location.host
          }))
      }
    })
    this.socketMessageHandler = this.$socketClient.onMessage
  },

  mounted() {
    this.canvas = this.$refs.bugCanvas
    this.ctx = this.canvas.getContext('2d')
    this.backgroundCanvas = this.$refs.backgroundCanvas
    this.backgroundCtx = this.backgroundCanvas.getContext('2d')
    this.initDrawing()
    this.prepareBackground().then(() => {
      if (!this.isDisposed) {
        this.initDrawing()
      }
    }).catch(error => console.error(error))
  },

  beforeDestroy() {
    this.isDisposed = true
    this.clearBoard()
    if (this.finishHoleImage) {
      this.finishHoleImage(false)
    }
    if (this.holeImage) {
      this.holeImage.onload = null
      this.holeImage.onerror = null
      this.holeImage.src = ''
      this.holeImage = null
    }
    if (this.$socketClient.onMessage === this.socketMessageHandler) {
      this.$socketClient.onMessage = null
    }
    if (this.$socketClient.onOpen === this.socketOpenHandler) {
      this.$socketClient.onOpen = null
    }
    if (this.$socketClient.onClose === this.socketCloseHandler) {
      this.$socketClient.onClose = null
    }
  },

  methods: {
    csvFlag(value) {
      return value === true || value === 1 || (typeof value === 'string' && ['true', '1'].includes(value.trim().toLowerCase()))
    },

    socketIsOpen() {
      return this.$socketClient.instance && this.$socketClient.instance.readyState === 1 // WebSocket.OPEN.
    },

    publish(channel, payload) {
      if (!this.socketIsOpen()) {
        return false
      }
      this.$socketClient.publish(channel, payload)
      return true
    },

    setVisualAppOff() {
      if (this.socketIsOpen()) {
        this.$socketClient.set('IS_VISUAL_APP_ON', 0)
      }
    },

    async initBoard(payload) {
      let options
      try {
        options = typeof payload === 'string' ? JSON.parse(payload) : payload
      } catch (error) {
        console.error('Invalid custom bugs payload:', error.message)
        return false
      }
      if (!options || (options.movementType && options.movementType !== 'custom') || !Array.isArray(options.trajectories)) {
        console.warn('Custom board ignored a payload without custom trajectories')
        return false
      }
      this.clearBoard()
      const token = this.trialToken
      const settings = defaultSettings()
      Object.keys(options).forEach(key => {
        if (options[key] === null || typeof options[key] !== 'object') {
          settings[key] = options[key] // Keep input trajectories and other large objects out of reactive settings.
        }
      })
      const validatedKeys = ['bugFps', 'bloodDuration', 'hitRadiusScale', 'holesHeightScale', 'holeSize']
      validatedKeys.forEach(key => {
        if (Object.prototype.hasOwnProperty.call(options, key)) {
          settings[key] = options[key] // Validate these supplied values even when they are not scalars.
        }
      })
      settings.movementType = 'custom'
      settings.numOfBugs = options.trajectories.length
      settings.isLogTrajectory = this.csvFlag(settings.isLogTrajectory)
      settings.isKillingAllByOneHit = this.csvFlag(settings.isKillingAllByOneHit)
      try {
        if (options.trajectories.length === 0) {
          throw new Error('Custom trial must contain at least one trajectory')
        }
        if (!['number', 'string'].includes(typeof settings.bugFps) || (settings.bugFps !== 'native' && (!Number.isFinite(Number(settings.bugFps)) || Number(settings.bugFps) <= 0))) {
          throw new Error('bugFps must be "native" or a positive number')
        }
        if (!Number.isFinite(Number(settings.bloodDuration)) || Number(settings.bloodDuration) < 0) {
          throw new Error('bloodDuration must be a nonnegative number')
        }
        if (!Number.isFinite(Number(settings.hitRadiusScale)) || Number(settings.hitRadiusScale) <= 0) {
          throw new Error('hitRadiusScale must be a positive number')
        }
        if (!Array.isArray(settings.holeSize) || settings.holeSize.length !== 2 || settings.holeSize.some(size => !Number.isFinite(Number(size)) || Number(size) <= 0)) {
          throw new Error('holeSize must contain two positive finite dimensions')
        }
        if (!Number.isFinite(Number(settings.holesHeightScale))) {
          throw new Error('holesHeightScale must be finite')
        }
        settings.holeSize = settings.holeSize.map(Number)
        settings.holesHeightScale = Number(settings.holesHeightScale)
        this.bugsSettings = settings
        this.bugsProps = options.trajectories.map((trajectory, index) => {
          if (!trajectory || !Array.isArray(trajectory.rows) || trajectory.rows.length === 0) {
            throw new Error('Each custom trajectory must contain at least one row')
          }
          return Object.freeze({
            slotIndex: index,
            slotKey: 'custom-' + token + '-' + index,
            trajectory: Object.freeze(trajectory) // Vue does not observe the CSV rows through this frozen object.
          })
        })
        this.trajectoryBugCount = this.bugsProps.length
        this.$store.commit('reset_score')
        await this.$nextTick()
        if (this.trialToken !== token || this.isDisposed) {
          return false
        }
        const bugs = this.$refs.bugChild || []
        if (bugs.length !== this.bugsProps.length) {
          throw new Error('Custom bug components did not mount')
        }
        this.activeBugs = bugs.slice()
        // await Promise.all([this.prepareBackground(), ...bugs.map(bug => bug.prepare(this.ctx))])
        // if (this.trialToken !== token || this.isDisposed) {
        //   return false
        // }

        const pool = {requests: new Map(), pending: new Set()}
        this.trialImagePool = pool

        await Promise.all([
          this.prepareBackground(),
          ...bugs.map(bug => bug.prepare(this.ctx, pool))
        ])

        pool.requests.clear() // Each bug now holds the images it needs.
        this.initDrawing()
        this.dumpTrialData()
        bugs.forEach(bug => bug.startPlayback())
        this.animate(performance.now())
        if (this.trialToken === token) {
          this.drawSquareForPhotoDiode() // Pulse only after images are ready and the first frame is drawn.
        }
        return this.trialToken === token
      } catch (error) {
        if (this.trialToken === token && !this.isDisposed) {
          console.error('Custom bugs could not start:', error.message)
          this.bugsSettings = defaultSettings()
          this.clearBoard()
          this.setVisualAppOff()
          this.publish('cmd/visual_app/console', 'Custom bugs could not start: ' + error.message)
        }
        return false
      }
    },

    clearBoard() {
      this.trialToken++ // Invalidate pending image preparation and callbacks from the previous trial.
      if (this.trialImagePool) {
        this.trialImagePool.pending.forEach(cancel => cancel())
        this.trialImagePool.requests.clear()
        this.trialImagePool = null
      }
      
      cancelAnimationFrame(this.animationHandler)
      this.animationHandler = null
      const timers = ['bloodTimeout', 'touchTimeout', 'climbingTimeout', 'rewardTimeout', 'squareTimeout']
      timers.forEach(key => {
        clearTimeout(this[key])
        this[key] = null
      })
      const bugs = new Set([...this.activeBugs, ...(this.$refs.bugChild || [])])
      bugs.forEach(bug => bug.stopPlayback())
      this.activeBugs = []
      this.bugsProps = []
      this.trialData = {}
      this.bugTrajectoryLog = []
      this.bugTrajectoryAnimationLog = []
      this.eventsLog = []
      this.trajectoryBugCount = 0
      this.touchesCounter = 0
      this.isBloodOnScreen = false
      this.isHandlingTouch = false
      this.isClimbing = false
      this.isRewardGiven = false
      if (this.ctx) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
      }
      this.initDrawing()
    },

    animate(now = performance.now()) {
      if (this.isDisposed || !this.trialData.start_time || this.trialData.end_time || this.activeBugs.length === 0) {
        return
      }
      const token = this.trialToken
      this.animationHandler = null
      try {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
        for (const bug of this.activeBugs) {
          if (this.trialToken !== token) {
            return
          }
          bug.move(this.activeBugs, now)
        }
        if (this.trialToken === token) {
          this.logAnimationTrajectory()
          this.animationHandler = requestAnimationFrame(this.animate)
        }
      } catch (error) {
        console.error('Custom animation stopped:', error.message)
        this.eventsLog.push({time: Date.now(), event: 'animation_error', message: error.message})
        this.endTrial()
      }
    },

    setCanvasClick(event) {
      const rect = this.canvas.getBoundingClientRect()
      const x = (event.clientX - rect.left) * this.canvas.width / rect.width
      const y = (event.clientY - rect.top) * this.canvas.height / rect.height
      this.handleTouchEvent(x, y)
      this.analyzeScreenTouch()
    },

    analyzeScreenTouch() {
      if (!this.trialData.start_time || this.trialData.end_time) {
        return
      }
      if (this.touchesCounter === 0) {
        this.touchTimeout = setTimeout(() => {
          this.touchTimeout = null
          this.touchesCounter = 0
        }, 4000)
      }
      this.touchesCounter++
      if (this.touchesCounter > 5 && !this.isClimbing) {
        this.eventsLog.push({time: Date.now(), event: 'climbing'})
        this.isClimbing = true
        this.climbingTimeout = setTimeout(() => {
          this.climbingTimeout = null
          this.isClimbing = false
        }, 10000)
        this.publish('cmd/visual_app/console', `screen climbing detected on trial #${this.bugsSettings.trialID}`)
      }
    },

    handleTouchEvent(x, y) {
      if (this.isHandlingTouch || !this.trialData.start_time || this.trialData.end_time || this.activeBugs.length === 0) {
        return
      }
      this.isHandlingTouch = true
      try {
        const distances = {}
        const hits = {}
        this.activeBugs.forEach((bug, index) => {
          if (!bug.isDead && !bug.isRetreated) {
            const candidate = [bug.hitDistance(x, y), bug.isHit(x, y), bug.isRewardBug]
            distances[index] = candidate
            if (candidate[1]) {
              hits[index] = candidate
            }
          }
        })
        const candidates = Object.keys(hits).length > 0 ? hits : distances // An unstrikeable overlapping bug cannot hide a valid hit.
        if (Object.keys(candidates).length === 0) {
          return
        }
        const index = Number(getKeyWithMinFirstArrayValue(candidates))
        const bug = this.activeBugs[index]
        let isHit = candidates[index][1]
        const isRewardBug = candidates[index][2]
        if (this.isClimbing || this.isBloodOnScreen) {
          this.eventsLog.push({
            time: Date.now(),
            event: 'touch_blocked',
            x,
            y,
            bug_id: bug.bugId,
            is_hit: isHit,
            is_reward_bug: isRewardBug,
            is_reward_any_touch: false,
            reason: this.isClimbing ? 'climbing' : 'blood_feedback'
          })
          isHit = false
        }
        if (isHit) {
          this.destruct(index, x, y, isRewardBug)
        }
        this.drawSquareForPhotoDiode()
        this.logTouch(x, y, bug, isHit, isRewardBug)
      } finally {
        this.isHandlingTouch = false
      }
    },

    logTouch(x, y, bug, isHit, isRewardBug) {
      this.publish('log/metric/touch', JSON.stringify({
        time: Date.now(),
        x,
        y,
        bug_x: bug.x,
        bug_y: bug.y,
        is_hit: isHit,
        is_reward_any_touch: false,
        is_reward_bug: isRewardBug,
        is_climbing: this.isClimbing,
        bug_type: this.getBugTypeForLog(bug),
        bug_size: bug.currentBugSize,
        in_block_trial_id: this.bugsSettings.trialID,
        trial_id: this.bugsSettings.trialDBId
      }))
    },

    destruct(bugIndex, x, y, isRewardBug) {
      const bug = this.activeBugs[bugIndex]
      if (!bug || bug.isDead || bug.isRetreated || this.isBloodOnScreen) {
        return
      }
      const token = this.trialToken
      const killAll = this.bugsSettings.isKillingAllByOneHit
      bug.isDead = true
      this.isBloodOnScreen = true
      if (isRewardBug) {
        this.$store.commit('increment')
      }
      this.bloodTimeout = setTimeout(() => {
        if (this.trialToken !== token) {
          return
        }
        this.bloodTimeout = null
        bug.stopPlayback()
        this.activeBugs = killAll ? [] : this.activeBugs.filter(item => item !== bug)
        this.isBloodOnScreen = false
        if (this.activeBugs.length === 0) {
          this.endTrial()
        }
      }, Number(this.bugsSettings.bloodDuration))
      if (killAll) {
        this.activeBugs.forEach(item => {
          if (item !== bug) {
            item.stopPlayback()
          }
        })
        this.activeBugs = [bug]
      }
    },

    endTrial() {
      if (this.bugsProps.length === 0 || this.trialData.end_time) {
        return
      }
      try {
        if (this.trialData.start_time) {
          const endTime = Date.now()
          this.trialData.end_time = endTime
          const payload = Object.assign({}, this.trialData, {
            duration: (endTime - this.trialData.start_time) / 1000,
            bug_trajectory: this.bugTrajectoryLog,
            bug_trajectory_animation: this.bugTrajectoryAnimationLog,
            app_events: this.eventsLog
          })
          if (!this.publish('log/metric/trial_data', JSON.stringify(payload))) {
            console.warn('Custom trial ended without a connection; trial metrics were not saved')
          }
        }
      } finally {
        this.clearBoard()
        this.setVisualAppOff()
        this.drawSquareForPhotoDiode()
      }
    },

    getBugTypeForLog(bug) {
      return typeof bug.getLoggedBugType === 'function' ? bug.getLoggedBugType() : bug.currentBugType
    },

    dumpTrialData() {
      this.trialData = {
        trial_db_id: this.bugsSettings.trialDBId,
        start_time: Date.now(),
        trial_bugs: this.activeBugs.map(bug => this.getBugTypeForLog(bug)).join(','),
        bug_sizes: this.activeBugs.map(bug => bug.currentBugSize).join(','),
        bug_speed: this.bugsSettings.speed,
        exit_hole: this.bugsSettings.exitHole,
        video_frames: null,
        movement_type: 'custom',
        extra: JSON.stringify({
          canvas_size: [this.canvas.width, this.canvas.height],
          bug_type: this.getBugTypeForLog(this.activeBugs[0]),
          bug_types: this.activeBugs.map(bug => this.getBugTypeForLog(bug))
        })
      }
    },

    recordCustomTrajectory(sample, slotKey) {
      if (!this.bugsSettings.isLogTrajectory || !this.trialData.start_time || this.trialData.end_time) {
        return
      }
      if (slotKey && !this.bugsProps.some(slot => slot.slotKey === slotKey)) {
        return
      }
      if (sample.trial_id === this.bugsSettings.trialDBId && this.activeBugs.some(bug => bug.bugId === sample.bug_id)) {
        this.bugTrajectoryLog.push(sample)
      }
    },

    logAnimationTrajectory() {
      if (!this.bugsSettings.isLogTrajectory || this.activeBugs.length === 0) {
        return
      }
      const entry = {
        time: Date.now(),
        in_block_trial_id: this.bugsSettings.trialID,
        trial_id: this.bugsSettings.trialDBId
      }
      for (let index = 0; index < Math.max(this.trajectoryBugCount, 2); index++) {
        entry[`bug${index}_x`] = null
        entry[`bug${index}_y`] = null
      }
      this.activeBugs.forEach(bug => {
        if (!bug.isDead && !bug.isRetreated) {
          entry[`bug${bug.bugId}_x`] = bug.x
          entry[`bug${bug.bugId}_y`] = bug.y
        }
      })
      this.bugTrajectoryAnimationLog.push(entry)
    },

    prepareBackground() {
      if (this.holeImagePromise) {
        return this.holeImagePromise
      }
      this.holeImagePromise = new Promise((resolve, reject) => {
        const image = new Image()
        this.holeImage = image
        this.finishHoleImage = resolve
        image.onload = () => {
          this.finishHoleImage = null
          image.onload = null
          image.onerror = null
          resolve(true)
        }
        image.onerror = () => {
          this.finishHoleImage = null
          image.onload = null
          image.onerror = null
          reject(new Error('Could not load the holes image'))
        }
        image.src = require('@/assets/hole2.png')
      })
      return this.holeImagePromise
    },

    initDrawing() {
      if (!this.backgroundCtx) {
        return
      }
      const canvas = this.backgroundCanvas
      const ctx = this.backgroundCtx
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      if (!this.holeImage || !this.holeImage.naturalWidth) {
        return
      }
      const [holeW, holeH] = this.bugsSettings.holeSize
      const heightOffset = (canvas.height - holeH / 2) * this.bugsSettings.holesHeightScale
      const y = canvas.height - holeH - heightOffset
      ctx.drawImage(this.holeImage, 100, y, holeW, holeH)
      ctx.drawImage(this.holeImage, canvas.width - holeW - 100, y, holeW, holeH)
    },

    drawSquareForPhotoDiode() {
      if (!this.backgroundCtx || this.isDisposed) {
        return
      }
      clearTimeout(this.squareTimeout)
      this.initDrawing()
      this.backgroundCtx.fillStyle = 'black'
      this.backgroundCtx.fillRect(0, 0, 50, 50)
      const token = this.trialToken
      this.squareTimeout = setTimeout(() => {
        this.squareTimeout = null
        if (this.trialToken === token && !this.isDisposed) {
          this.initDrawing()
        }
      }, 100)
    }
  }
}
</script>

<style scoped>
#custom-board {
  position: fixed;
  top: 0;
  left: 0;
  width: 100vw;
  height: 100vh;
}

#customBackgroundCanvas,
#customBugCanvas {
  position: absolute;
  top: 0;
  left: 0;
  display: block;
}

#customBugCanvas {
  z-index: 1;
}
</style>