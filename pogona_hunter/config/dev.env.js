'use strict'
const merge = require('webpack-merge')
const prodEnv = require('./prod.env')

module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  NUM_BUGS: 2,
  IS_SPLIT_BUGS_VIEW: false,
  IS_SPLIT_MIRROR: true,
  ROUTER_MODE: '"history"',
  MOVEMENT_TYPE: '"low_horizontal"', //low_horizontal  //circle
  BUG_TYPES: '["green_cockroach", "red_cockroach"]',
  REWARD_BUGS: '"green_cockroach"'
})
