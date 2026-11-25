import {randomRange, getKeyWithMinFirstArrayValue} from '../../js/helpers'
import { v4 as uuidv4 } from 'uuid'

export default {

  data() {
    return {
      configOptions: require('../../config.json'),
      bugsProps: [],
      trajectories: [],
      bugsSettings: {
        numOfBugs: process.env.NUM_BUGS,
        isSplitBugsView: process.env.IS_SPLIT_BUGS_VIEW,
        trialID: null,
        trialDBId: null,
        numTrials: null, // deprecated. Trials are governed by the experiment
        trialDuration: 5,
        iti: 5,
        bugTypes: process.env.BUG_TYPES || [],
        rewardBugs: process.env.REWARD_BUGS || [],
        movementType: process.env.MOVEMENT_TYPE || 'circle',
        speed: 0, // if 0 config default for bug will be used
        bugSize: 0, // if 0 will be handled in bug component (native size or config default)
        bloodDuration: 2000,
        backgroundColor: '#e8eaf6',
        bugMappedBackground: {},
        rewardAnyTouchProb: 0,
        hitRadiusScale: 2 / 3,
        accelerateMultiplier: 3, // times to increase bug speed in tongue detection
        preTunnelSpeedMultiplier: 1, // multiplier for tunnel entry acceleration
        isKillingAllByOneHit: process.env.IS_KILLING_ALL_BY_ONE_HIT, // if true, all bugs will disapear when one is hit successfully
        splitRandomizeTiming: process.env.SPLIT_RANDOMIZE_TIMING, // if 1, randomize timing of bugs appearance
        bugFps: 60,
        isLogTrajectory: false
      },
      mediaUrl: '',
      isHandlingTouch: false,
      isRewardGiven: false,
      isClimbing: false,
      isBloodOnScreen: false, // bug is with blood on screen so avoid any other hits until blood goes away
      bloodTimeout: null,
      afterRewardTimeout: 40 * 1000,
      touchesCounter: 0,
      canvasParams: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      bugTrajectoryLog: [],
      bugTrajectoryAnimationLog: [],
      trajectoryBugCount: null,
      trialData: {},
      eventsLog: [],
      identifier: uuidv4()
    }
  },

  created() {
    this.$socketClient.onOpen = () => {
      console.log('WebSocket connected')
    }
    this.$socketClient.subscribe({
      'cmd/visual_app/hide_bugs': (payload) => {
        console.log('received hide_bugs command')
        if (this.isCustomBugs) {
          this.endTrial() // Stop custom playback both during preparation and after startup.
          return
        }
        if (this.$refs.bugChild) {
          this.$refs.bugChild.forEach(bug => bug.startRetreat())
        }
      },
      'cmd/visual_app/init_bugs': (options) => {
        options = JSON.parse(options)
        const {trajectories = [], ...settings} = options
        this.trajectories = trajectories
        Object.assign(this.bugsSettings, settings)
        console.log('Received init_bugs options:', options)
        this.$store.commit('reset_score')
        this.initBoard(!!options['isLogTrajectory'])
      },
      'cmd/visual_app/strike_predicted': (payload) => {
        console.log('received strike_predicted command')
        this.jumpBugs()
        this.eventsLog.push({
          time: Date.now(),
          event: 'bug_jump'
        })
      },
      'cmd/visual_app/flip_circle_direction': (payload) => {
        console.log('received flip_circle_direction command')
        this.flipCircle()
        this.eventsLog.push({
          time: Date.now(),
          event: 'flip_circle'
        })
      },
      'cmd/visual_app/reload_app': (payload) => {
        location.reload()
      },
      'cmd/visual_app/reward_given': (payload) => {
        console.log('reward was given in the arena')
        this.isRewardGiven = true
        let rewardTimeout = setTimeout(() => {
          this.isRewardGiven = false
          clearTimeout(rewardTimeout)
        }, 20 * 1000)
      },
      'cmd/visual_app/healthcheck': () => {
        this.$socketClient.publish('cmd/visual_app/healthcheck', JSON.stringify({
          id: this.identifier, host: location.host
        }))
      }
    })
  },

  mounted() {
    this.canvas = document.getElementById('bugCanvas')
    this.ctx = this.canvas.getContext('2d')
    this.initBoard()
    window.addEventListener('keypress', e => {
      if (e.code === 'KeyJ') {
        this.jumpBugs()
      }
    })
    window.addEventListener('keypress', e => {
      if (e.code === 'KeyF') {
        this.flipCircle()
      }
    })
  },

  computed: {

    isCustomBugs: function () {
      return this.bugsSettings.movementType === 'custom'
    },

    isSplitBugsView: function () {
      return this.bugsSettings.isSplitBugsView && this.bugsSettings.numOfBugs > 1
    },

    currentBugType: function () {
      const bugs = this.$refs.bugChild || []
      return bugs.length > 0 ? bugs[0].currentBugType : null
    }
  },

  methods: {

    initBoard(isLogTrajectory = false) {
      const previousBugs = this.$refs.bugChild || []
      this.bugsProps = []
      cancelAnimationFrame(this.animationHandler)
      this.animationHandler = null
      clearTimeout(this.bloodTimeout)
      this.bloodTimeout = null
      this.releaseBoardBugs(previousBugs)
      this.$refs.bugChild = []
      this.endLogBugTrajectory()
      this.trialData = {}
      this.eventsLog = []
      this.isBloodOnScreen = false
      this.isHandlingTouch = false
      this.bugsSettings.isLogTrajectory = isLogTrajectory
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
      this.bugsSettings.bugTypes = this.normalizeBugTypes(this.bugsSettings.bugTypes)
      const hasActiveBugConfig = this.isCustomBugs
        ? this.trajectories.length > 0
        : this.hasActiveBugConfig(this.bugsSettings.bugTypes)
      if (!hasActiveBugConfig) {
        console.log('No active bug payload; keeping board idle until init_bugs arrives')
        this.initDrawing()
        if (this.$socketClient.instance && this.$socketClient.instance.readyState === WebSocket.OPEN) {
          this.$socketClient.set('IS_VISUAL_APP_ON', 0) // The initial page mount can precede the socket connection.
        }
        return
      }
      if (!this.isCustomBugs && this.bugsSettings.bugTypes.length === 0) {
        console.warn('No bug types received for trial; defaulting to worm')
        this.bugsSettings.bugTypes = ['worm']
      }
      this.initDrawing()
      this.drawSquareForPhotoDiode()
      let bugsToSpawn = Number(this.bugsSettings.numOfBugs)
      bugsToSpawn = Number.isFinite(bugsToSpawn) && bugsToSpawn > 0 ? bugsToSpawn : 1
      if (this.isSplitBugsView) {
        const baseType = this.bugsSettings.bugTypes[0]
        let types = (this.bugsSettings.bugTypes.length > 1)
          ? [...this.bugsSettings.bugTypes]
          : Array(bugsToSpawn).fill(baseType)
        if (this.bugsSettings.exitHole === 'right') {
          types.reverse()
        }
        this.bugsSettings.bugTypes = types
        bugsToSpawn = Math.max(bugsToSpawn, this.bugsSettings.bugTypes.length)
      }
      if (this.isCustomBugs) {
        const trialKey = uuidv4()
        this.bugsProps = this.trajectories.map((trajectory, index) => ({
          slotIndex: index,
          slotKey: 'custom-' + trialKey + '-' + index
        }))
      } else {
        this.spawnBugs(bugsToSpawn)
      }
      this.trajectoryBugCount = this.bugsProps.length || null
      const trialBugs = this.bugsProps
      this.$nextTick(async function () {
        if (this.bugsProps !== trialBugs) {
          return
        }
        const bugs = this.$refs.bugChild || []
        if (bugs.length === 0) {
          this.endTrial()
          return
        }
        if (this.isCustomBugs) {
          const ready = await this.prepareCustomBugs(bugs, trialBugs, isLogTrajectory)
          if (!ready || this.bugsProps !== trialBugs) {
            return
          }
        }
        if (isLogTrajectory && !this.isCustomBugs) {
          this.startLogBugTrajectory()
        }
        this.dumpTrialData()
        this.startBoardBugs(bugs)
        this.animate()
      })
    },
    initDrawing() {
    },
    clearBoard() {
      const bugs = this.$refs.bugChild || []
      this.bugsProps = [] // Invalidate pending preparation and callbacks.
      cancelAnimationFrame(this.animationHandler)
      this.animationHandler = null
      clearTimeout(this.bloodTimeout)
      this.bloodTimeout = null
      this.releaseBoardBugs(bugs)
      this.$refs.bugChild = []
      this.bugsSettings.numOfBugs = 0
      this.bugsSettings.trialID = null
      this.bugsSettings.trialDBId = null
      this.isBloodOnScreen = false
      this.endLogBugTrajectory()
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
    },
    animate(now = performance.now()) {
      if (!this.$refs.bugChild || this.bugsProps.length === 0 || !this.trialData.start_time || this.trialData.end_time) {
        return
      }
      const trialBugs = this.bugsProps
      const bugs = this.$refs.bugChild
      try {
        this.animationHandler = requestAnimationFrame(this.animate)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
        for (const bug of bugs) {
          if (this.bugsProps !== trialBugs) {
            return
          }
          bug.move(bugs, now) // One move per frame, with a shared timestamp.
        }
        if (this.bugsProps === trialBugs) {
          this.logAnimationTrajectory()
        }
      } catch (e) {
        console.error(e)
        cancelAnimationFrame(this.animationHandler)
        this.animationHandler = null
      }
    },

    jumpBugs() {
      if (this.$refs.bugChild) {
        this.$refs.bugChild.forEach(bug => bug.jump())
      }
    },
    flipCircle() {
      if (this.$refs.bugChild) {
        this.$refs.bugChild.forEach(bug => bug.flipCircle())
        this.drawSquareForPhotoDiode()
      }
    },

    async prepareCustomBugs(bugs, trialBugs, isLogTrajectory) {
      try {
        await this.prepareBoardBugs(bugs)
      } catch (error) {
        if (this.bugsProps !== trialBugs) {
          return false
        }
        console.error(error)
        this.clearBoard()
        this.$socketClient.set('IS_VISUAL_APP_ON', 0)
        this.$socketClient.publish(
        'cmd/visual_app/console',
        'Custom bugs could not start: ' + error.message
        )
        return false
      }
      if (this.bugsProps !== trialBugs) {
        return false
      }
      this.trajectoryBugCount = bugs.length
      this.bugsSettings.isLogTrajectory = isLogTrajectory
      return true
    },

    prepareBoardBugs() {},

    startBoardBugs() {},

    releaseBoardBugs() {},

    getBugRewardStatus(bug) {
      const rewardBugs = this.bugsSettings.rewardBugs || []
      return Array.isArray(rewardBugs)
        ? rewardBugs.includes(bug.currentBugType)
        : String(rewardBugs).includes(bug.currentBugType)
    },

    getRewardAnyTouch() {
      return Math.random() < this.bugsSettings.rewardAnyTouchProb
    },

    filterTouchCandidates(strikeDistances) {
      return strikeDistances
    },

    setCanvasTouch(event) {
      for (let touch of event.touches) {
        this.handleTouchEvent(touch.screenX, touch.screenY)
      }
    },
    setCanvasClick(event) {
      this.handleTouchEvent(event.x, event.y)
    },
    analyzeScreenTouch(event) {
      if (this.touchesCounter === 0) {
        let that = this
        let touchesCounterTimeout = setTimeout(() => {
          that.touchesCounter = 0
          clearTimeout(touchesCounterTimeout)
        }, 4000)
      }
      this.touchesCounter++
      if (this.touchesCounter > 5 && !this.isClimbing) {
        console.log('climbing!')
        this.eventsLog.push({
            time: Date.now(),
            event: 'climbing'
        })
        this.isClimbing = true
        let climbingTimout = setTimeout(() => {
          this.isClimbing = false
          clearTimeout(climbingTimout)
        }, 10000)
        this.$socketClient.publish('cmd/visual_app/console',
          `screen climbing detected on trial #${this.bugsSettings.trialID}`)
      }
    },
    handleTouchEvent(x, y) {
      console.log(x, y)
      if (this.isHandlingTouch || !this.$refs.bugChild || !this.trialData.start_time || this.trialData.end_time) {
        return
      }
      this.isHandlingTouch = true
      try {
        x -= this.canvas.offsetLeft
        y -= this.canvas.offsetTop
        let strikeDistances = {}
        let isRewardAnyTouch = this.getRewardAnyTouch()
        for (let i = 0; i < this.$refs.bugChild.length; i++) {
          let bug = this.$refs.bugChild[i]
          if (bug.isDead || bug.isRetreated) {
            continue
          }
          let isRewardBug = this.getBugRewardStatus(bug)
          strikeDistances[i] = [bug.hitDistance(x, y), bug.isHit(x, y), isRewardBug]
        }
        strikeDistances = this.filterTouchCandidates(strikeDistances)
        if (Object.keys(strikeDistances).length > 0) {
          let i = Number(getKeyWithMinFirstArrayValue(strikeDistances))
          let bug = this.$refs.bugChild[i]
          let isHit = strikeDistances[i][1]
          let isRewardBug = strikeDistances[i][2]
          if (this.isClimbing || this.isBloodOnScreen) {
            this.eventsLog.push({
              time: Date.now(),
              event: 'touch_blocked',
              x,
              y,
              bug_id: bug.bugId,
              is_hit: isHit,
              is_reward_bug: isRewardBug,
              is_reward_any_touch: isRewardAnyTouch,
              reason: this.isClimbing ? 'climbing' : 'blood_feedback'
            })
            isHit = false // Prevent a blocked touch being published as a rewarded hit.
            isRewardAnyTouch = false
          }
          if (isHit || isRewardAnyTouch) {
            this.destruct(i, x, y, isRewardBug)
          }
          this.drawSquareForPhotoDiode()
          this.logTouch(x, y, bug, isHit, isRewardBug, isRewardAnyTouch)
        }
      } finally {
        this.isHandlingTouch = false
      }
    },

    logTouch(x, y, bug, isHit, isRewardBug, isRewardAnyTouch) {
      this.$socketClient.publish('log/metric/touch', JSON.stringify({
          time: Date.now(),
          x: x,
          y: y,
          bug_x: bug.x,
          bug_y: bug.y,
          is_hit: isHit,
          is_reward_any_touch: isRewardAnyTouch,
          is_reward_bug: isRewardBug,
          is_climbing: this.isClimbing,
          bug_type: bug.currentBugType,
          bug_size: bug.currentBugSize,
          in_block_trial_id: this.bugsSettings.trialID,
          trial_id: this.bugsSettings.trialDBId
        }))
      console.log('Touch event was sent to the server')
    },
    destruct(bugIndex, x, y, isRewardBug) {
      const bugs = this.$refs.bugChild || []
      const bug = bugs[bugIndex]
      if (!bug || bug.isDead || bug.isRetreated || this.isBloodOnScreen) {
        return
      }
      const trialBugs = this.bugsProps
      const killAll = this.bugsSettings.isKillingAllByOneHit
      bug.isDead = true
      this.isBloodOnScreen = true
      if (isRewardBug) {
        this.$store.commit('increment')
      }
      this.bloodTimeout = setTimeout(() => {
        if (this.bugsProps !== trialBugs) {
          return // An old timeout must not affect the next trial.
        }
        this.bloodTimeout = null
        this.releaseBoardBugs([bug])
        this.$refs.bugChild = killAll
          ? []
          : (this.$refs.bugChild || []).filter(item => item !== bug)
        this.isBloodOnScreen = false
        if (this.$refs.bugChild.length === 0) {
          this.endTrial()
        }
      }, this.bugsSettings.bloodDuration)
      if (killAll) {
        this.releaseBoardBugs(bugs.filter(item => item !== bug))
        this.$refs.bugChild = [bug]
      }
    },
    endTrial() {
      if (this.bugsProps.length === 0 || this.trialData.end_time) {
        return
      }
      const trialID = this.bugsSettings.trialID
      try {
        if (this.trialData.start_time) {
          let endTime = Date.now()
          let payload = {
            duration: (endTime - this.trialData.start_time) / 1000,
            end_time: endTime,
            bug_trajectory: this.bugTrajectoryLog,
            bug_trajectory_animation: this.bugTrajectoryAnimationLog,
            app_events: this.eventsLog
          }
          this.trialData = Object.assign(this.trialData, payload)
          this.$socketClient.publish('log/metric/trial_data', JSON.stringify(this.trialData))
          console.log(`Trial ${trialID} data was sent to the server`)
        }
      } finally {
        this.clearBoard()
        this.$socketClient.set('IS_VISUAL_APP_ON', 0)
        this.drawSquareForPhotoDiode()
      }
    },
    dumpTrialData() {
      this.trialData = {
        trial_db_id: this.bugsSettings.trialDBId,
        start_time: Date.now(),
        trial_bugs: this.$refs.bugChild.map(bug => bug.currentBugType).join(','),
        bug_sizes: this.$refs.bugChild.map(bug => bug.currentBugSize).join(','),
        bug_speed: this.bugsSettings.speed,
        exit_hole: this.bugsSettings.exitHole,
        video_frames: null,
        extra: JSON.stringify(this.extraTrialData())
      }
    },
    extraTrialData: function () {
      return {}
    },
    normalizeBugTypes(value) {
      if (Array.isArray(value)) {
        return value.map((v) => String(v).trim()).filter(Boolean)
      }
      if (typeof value === 'string') {
        return value.split(',').map((v) => v.trim()).filter(Boolean)
      }
      if (value === null || value === undefined) {
        return []
      }
      return [String(value).trim()].filter(Boolean)
    },
    hasActiveBugConfig(normalizedBugTypes = null) {
      const bugTypes = Array.isArray(normalizedBugTypes)
        ? normalizedBugTypes
        : this.normalizeBugTypes(this.bugsSettings.bugTypes)
      if (bugTypes.length > 0) {
        return true
      }
      const configuredCount = Number(this.bugsSettings.numOfBugs)
      return Number.isFinite(configuredCount) && configuredCount > 0
    },
    spawnBugs(noOfBugs) {
      for (let i = 0; i < noOfBugs; i++) {
        let x = randomRange(0, this.canvas.width)
        let y = randomRange(0, this.canvas.height)
        let properties = {
          x: x,
          y: y,
          bugId: `${this.bugsSettings.bugTypes[i]}_${i}`,
          slotIndex: i,
          slotKey: `bug-slot-${i}`
        }
        this.bugsProps.push(properties)
      }
    },
    startBugsEscape(event) {
      console.log('rightClick', event.x, event.y)
      this.$refs.bugChild.forEach(bug => bug.escape(event.x, event.y))
    },
    logAnimationTrajectory() {
      const bugs = this.$refs.bugChild || []
      const isLogging = this.isCustomBugs
        ? this.bugsSettings.isLogTrajectory
        : Boolean(this.trajectoryLogInterval)
      if (!isLogging || bugs.length === 0) {
        return
      }
      const entry = {
        time: Date.now(),
        in_block_trial_id: this.bugsSettings.trialID,
        trial_id: this.bugsSettings.trialDBId
      }
      for (let i = 0; i < Math.max(this.trajectoryBugCount || 0, 2); i++) {
        entry[`bug${i}_x`] = null
        entry[`bug${i}_y`] = null
      }
      bugs.forEach((bug, index) => {
        const id = Number.isInteger(bug.bugId) ? bug.bugId : index
        if (!bug.isDead && !bug.isRetreated) {
          entry[`bug${id}_x`] = bug.x
          entry[`bug${id}_y`] = bug.y
        }
      })
      this.bugTrajectoryAnimationLog.push(entry)
    },
    startLogBugTrajectory() {
      console.log('trajectory log started')
      if (!Number.isFinite(this.trajectoryBugCount) || this.trajectoryBugCount <= 0) {
        const fallbackCount = this.bugsProps ? this.bugsProps.length : 0
        this.trajectoryBugCount = fallbackCount > 0 ? fallbackCount : null
      }
      if (!this.trajectoryBugCount) {
        console.log('no bugs to log')
        return
      }
      this.trajectoryLogInterval = setInterval(() => {
        const bugs = this.$refs.bugChild || []
        if (bugs.length === 0) {
          console.log('no bugs to log')
          return
        }
        const entry = {
          time: Date.now(),
          in_block_trial_id: this.bugsSettings.trialID,
          trial_id: this.bugsSettings.trialDBId
        }
        const slotCount = Math.max(this.trajectoryBugCount || 0, 2)
        for (let i = 0; i < slotCount; i++) {
          entry[`bug${i}_x`] = null
          entry[`bug${i}_y`] = null
          entry[`bug${i}_type`] = null
          entry[`bug${i}_alive`] = null
        }
        const bugMap = {}
        bugs.forEach((bug, idx) => {
          let bugIndex = idx
          if (bug && Number.isFinite(bug.bugId)) {
            bugIndex = bug.bugId
          } else if (bug && typeof bug.bugId === 'string' && bug.bugId.trim() !== '' && !Number.isNaN(Number(bug.bugId))) {
            bugIndex = Number(bug.bugId)
          }
          if (!Number.isFinite(bugIndex) || bugIndex < 0) {
            bugIndex = idx
          }
          bugMap[bugIndex] = bug
        })
        for (let i = 0; i < slotCount; i++) {
          const bug = bugMap[i]
          if (bug) {
            entry[`bug${i}_type`] = bug.currentBugType || null
            const isAlive = !(bug.isDead || bug.isRetreated)
            entry[`bug${i}_alive`] = isAlive
            if (isAlive) {
              entry[`bug${i}_x`] = bug.x
              entry[`bug${i}_y`] = bug.y
            }
          }
        }
        this.bugTrajectoryLog.push(entry)
      }, 1000 / 60)
    },
    endLogBugTrajectory() {
      clearInterval(this.trajectoryLogInterval)
      this.trajectoryLogInterval = null
      this.bugTrajectoryLog = []
      this.bugTrajectoryAnimationLog = [] // Prevent animation samples carrying over into the next trial.
      this.trajectoryBugCount = null
      console.log('trajectory log ended')
    },
    getBugMappedBackgroundColor(bugType) {
      if (!this.bugsSettings.bugMappedBackground || !this.bugsSettings.bugMappedBackground[bugType]) {
        return this.bugsSettings.backgroundColor // fallback to default
      }
      return this.bugsSettings.bugMappedBackground[bugType]
    },
    getCurrentBackgroundColor() {
      if (!this.bugsSettings.bugMappedBackground) {
        return this.bugsSettings.backgroundColor
      }
      if (this.bugsSettings.bugTypes && this.bugsSettings.bugTypes.length > 0) {
        return this.getBugMappedBackgroundColor(this.bugsSettings.bugTypes[0])
      }
      return this.bugsSettings.backgroundColor
    },
    drawSplitBackground() {
      const canvas = document.getElementById('backgroundCanvas')
      const ctx = canvas.getContext('2d')
      const canvasWidth = canvas.width
      const canvasHeight = canvas.height
      ctx.clearRect(0, 0, canvasWidth, canvasHeight)
      if (!this.bugsSettings.bugMappedBackground || !this.bugsSettings.bugTypes || this.bugsSettings.bugTypes.length < 2) {
        ctx.fillStyle = this.getCurrentBackgroundColor()
        ctx.fillRect(0, 0, canvasWidth, canvasHeight)
        return
      }
      const leftBugType = this.bugsSettings.bugTypes[0]
      const rightBugType = this.bugsSettings.bugTypes[1]
      const leftColor = this.getBugMappedBackgroundColor(leftBugType)
      const rightColor = this.getBugMappedBackgroundColor(rightBugType)
      ctx.fillStyle = leftColor
      ctx.fillRect(0, 0, canvasWidth / 2, canvasHeight)
      ctx.fillStyle = rightColor
      ctx.fillRect(canvasWidth / 2, 0, canvasWidth / 2, canvasHeight)
    },
    drawSolidBackground(currentBugType) {
      const canvas = document.getElementById('backgroundCanvas')
      const ctx = canvas.getContext('2d')
      const color = this.getBugMappedBackgroundColor(currentBugType)
      ctx.fillStyle = color
      ctx.fillRect(0, 0, canvas.width, canvas.height)
    },
    drawSquareForPhotoDiode: function () {
      const canvas = document.getElementById('backgroundCanvas')
      const ctx = canvas.getContext('2d')
      const imageData = ctx.getImageData(0, 0, 50, 50) // Save the current background in that area
      ctx.fillStyle = 'black'
      ctx.fillRect(0, 0, 50, 50) // Draws a 100x100 black square at (0, 0)
      let squareTimeout = setTimeout(() => {
        ctx.putImageData(imageData, 0, 0)
        clearTimeout(squareTimeout)
      }, 100)
    }
  }
}