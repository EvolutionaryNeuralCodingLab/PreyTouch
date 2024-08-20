import {distance, randomRange} from '../../js/helpers'
import { v4 as uuidv4 } from 'uuid'

export default {
  data() {
    return {
      configOptions: require('../../config.json'),
      bugsProps: [],
      bugsSettings: {
        numOfBugs: process.env.NUM_BUGS,
        trialID: null,
        trialDBId: null,
        trialStartTime: null,
        numTrials: null, // deprecated. Trials are governed by the experiment
        trialDuration: 20,
        iti: 5,
        bugTypes: ['cockroach'],
        rewardBugs: 'cockroach',
        movementType: 'jump_up',
        speed: 0, // if 0 config default for bug will be used
        bugSize: 0, // if 0 config default for bug will be used
        bloodDuration: 2000,
        backgroundColor: '#e8eaf6',
        rewardAnyTouchProb: 0
      },
      mediaUrl: '',
      isHandlingTouch: false,
      isRewardGiven: false,
      isClimbing: false,
      afterRewardTimeout: 40 * 1000,
      touchesCounter: 0,
      canvasParams: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      bugTrajectoryLog: [],
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
        if (this.$refs.bugChild) {
          this.$refs.bugChild.forEach(bug => bug.startRetreat())
        }
      },
      'cmd/visual_app/init_bugs': (options) => {
        options = JSON.parse(options)
        console.log(options)
        // this.$socketClient.publish('cmd/visual_app/console', 'Trial started')
        Object.assign(this.bugsSettings, options)
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
  },
  computed: {
    currentBugType: function () {
      let bug = this.$refs.bugChild[0]
      return bug.currentBugType
    }
  },
  methods: {
    initBoard(isLogTrajectory = false) {
      if (this.animationHandler) {
        this.$refs.bugChild = []
        cancelAnimationFrame(this.animationHandler)
      }
      if (isLogTrajectory) {
        this.startLogBugTrajectory()
      }
      this.bugsSettings.trialStartTime = Date.now()
      this.initDrawing()
      this.spawnBugs(this.bugsSettings.numOfBugs)
      this.$nextTick(function () {
        console.log('start animation...')
        this.animate()
      })
    },
    initDrawing() {
      // drawing that should occur at board init
    },
    clearBoard() {
      this.bugsSettings.numOfBugs = 0
      if (this.animationHandler) {
        this.$refs.bugChild = []
        cancelAnimationFrame(this.animationHandler)
      }
      this.bugsSettings.trialID = null
      this.bugsSettings.trialDBId = null
      this.$nextTick(function () {
        console.log('Clear board')
        this.animate()
      })
    },
    animate() {
      if (!this.$refs.bugChild) {
        return
      }
      try {
        this.animationHandler = requestAnimationFrame(this.animate)
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height)
        this.$refs.bugChild.forEach(bug => bug.move(this.$refs.bugChild))
      } catch (e) {
        console.log(e)
        cancelAnimationFrame(this.animationHandler)
      }
    },
    jumpBugs() {
      if (this.$refs.bugChild) {
          this.$refs.bugChild.forEach(bug => bug.jump())
        }
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
      if (this.isHandlingTouch || !this.$refs.bugChild) {
        return
      }
      this.isHandlingTouch = true
      x -= this.canvas.offsetLeft
      y -= this.canvas.offsetTop
      for (let i = 0; i < this.$refs.bugChild.length; i++) {
        let bug = this.$refs.bugChild[i]
        if (bug.isDead || bug.isRetreated) {
          continue
        }
        let isRewardBug = this.bugsSettings.rewardBugs.includes(bug.currentBugType)
        let isHit = bug.isHit(x, y)
        let isRewardAnyTouch = Math.random() < this.bugsSettings.rewardAnyTouchProb
        if ((isHit || isRewardAnyTouch) && !this.isClimbing) {
          this.destruct(i, x, y, isRewardBug)
        }
        this.logTouch(x, y, bug, isHit, isRewardBug, isRewardAnyTouch)
      }
      this.isHandlingTouch = false
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
      let currentBugs = this.$refs.bugChild
      let lastBugType = currentBugs[bugIndex].currentBugType
      currentBugs[bugIndex].isDead = true
      if (isRewardBug) {
        this.$refs.audio1.play()
        this.$store.commit('increment')
      }
      const bloodTimeout = setTimeout(() => {
        this.$refs.bugChild = currentBugs.filter((items, index) => bugIndex !== index)
        if (this.$refs.bugChild.length === 0) {
          this.endTrial(lastBugType)
        }
        clearTimeout(bloodTimeout)
      }, this.bugsSettings.bloodDuration)
    },
    endTrial(lastBugType) {
      // endTrial can be called only after: 1) bug caught [destruct method], 2) trial time reached
      let endTime = Date.now()
      let trialID = this.bugsSettings.trialID
      let payload = {
        trial_db_id: this.bugsSettings.trialDBId,
        start_time: this.bugsSettings.trialStartTime,
        bug_type: lastBugType,
        duration: (endTime - this.bugsSettings.trialStartTime) / 1000,
        end_time: endTime,
        bug_trajectory: this.bugTrajectoryLog,
        video_frames: null,
        app_events: this.eventsLog,
        exit_hole: this.bugsSettings.exitHole
      }
      this.$socketClient.set('IS_VISUAL_APP_ON', 0)
      this.clearBoard()
      this.endLogBugTrajectory()
      this.$socketClient.publish('log/metric/trial_data', JSON.stringify(payload))
      console.log(`Trial ${trialID} data was sent to the server`)
    },
    spawnBugs(noOfBugs) {
      const minDistance = 100
      for (let i = 0; i < noOfBugs; i++) {
        let x = randomRange(0, this.canvas.width)
        let y = randomRange(0, this.canvas.height)
        let properties = {
          x: x,
          y: y,
          bugId: `${this.bugsSettings.bugTypes}${i}`
        }
        if (i !== 0) {
          for (let j = 0; j < i; j++) {
            let d = distance(x, y, this.bugsProps[j].x, this.bugsProps[j].y)
            if (d <= minDistance) {
              x = randomRange(0, this.canvas.width)
              y = randomRange(0, this.canvas.height)
              j = -1
            }
          }
        }
        this.bugsProps.push(properties)
      }
    },
    startBugsEscape(event) {
      console.log('rightClick', event.x, event.y)
      this.$refs.bugChild.forEach(bug => bug.escape(event.x, event.y))
    },
    startLogBugTrajectory() {
      console.log('trajectory log started')
      this.trajectoryLogInterval = setInterval(() => {
        let bug = this.$refs.bugChild[0]
        if (bug) {
          this.bugTrajectoryLog.push({
            time: Date.now(),
            x: bug.x,
            y: bug.y
          })
        }
      }, 1000 / 60)
    },
    endLogBugTrajectory() {
      clearInterval(this.trajectoryLogInterval)
      this.trajectoryLogInterval = null
      this.bugTrajectoryLog = []
      console.log('trajectory log ended')
    }
  }
}
