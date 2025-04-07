'use strict'
const merge = require('webpack-merge')
const prodEnv = require('./prod.env')

module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  NUM_BUGS: 2,
  IS_SPLIT_BUGS_VIEW: true,
  ROUTER_MODE: '"history"',
  MOVEMENT_TYPE: '"circle"', //low_horizontal  //circle
  BUG_TYPES: '[ "red_cockroach", "green_cockroach"]',
  REWARD_BUGS: '"green_cockroach"',
  RANDOMIZE_TIMING: 1,
})
