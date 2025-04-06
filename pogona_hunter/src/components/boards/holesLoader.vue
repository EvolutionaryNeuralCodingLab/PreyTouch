<template>
  <component
    :is="boardComponent"
    :bugsSettings="mergedSettings"
    v-bind="$attrs"
    v-on="$listeners"
  />
</template>

<script>
import holesBoard from './holesBoard.vue'
import mirroredBoard from './mirroredBoard.vue'

export default {
  name: 'holesLoader',

  props: {
    bugsSettings: {
      type: Object,
      default: () => ({})
    }
  },

  computed: {
    defaultSettings() {
      return {
        numOfBugs: process.env.NUM_BUGS,
        isSplitBugsView: process.env.IS_SPLIT_BUGS_VIEW,
        trialID: null,
        trialDBId: null,
        numTrials: null, // deprecated. Trials are governed by the experiment
        trialDuration: 5,
        iti: 5,
        bugTypes: process.env.BUG_TYPES || ['cockroach', 'green_beetle'],
        rewardBugs: process.env.REWARD_BUGS || 'cockroach',
        movementType: process.env.MOVEMENT_TYPE || 'circle',
        speed: 0, // if 0 config default for bug will be used
        bugSize: 0, // if 0 config default for bug will be used
        bloodDuration: 2000,
        backgroundColor: '#e8eaf6',
        rewardAnyTouchProb: 0,
        accelerateMultiplier: 3, // times to increase bug speed in tongue detection
        isKillingAllByOneHit: process.env.IS_KILLING_ALL_BY_ONE_HIT, // if true, all bugs will disapear when one is hit successfully
        randomizeTiming: process.env.RANDOMIZE_TIMING || 1// 0 - no randomization, 1 - randomize between 0 and 1
      }
    },

    mergedSettings() {
      // Merge defaultSettings with passed bugsSettings and URL query params
      const querySettings = Object.fromEntries(
        Object.entries(this.$route.query).map(([k, v]) => [
          k,
          v === 'true' ? true : v === 'false' ? false : isNaN(v) ? v : Number(v)
        ])
      )

      return {
        ...this.defaultSettings,
        ...this.bugsSettings,
        ...querySettings
      }
    },

    boardComponent() {
      const component = this.mergedSettings.isSplitBugsView ? mirroredBoard : holesBoard
      console.log(`Loading ${component.name}`)
      return component
    }
  }
}
</script>
