'use strict'
const merge = require('webpack-merge')
const prodEnv = require('./prod.env')

module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  NUM_BUGS: 2,
  ROUTER_MODE: '"history"',
  MOVEMENT_TYPE: '"low_horizontal"',
  BUG_TYPES: '["green_cockroach", "red_cockroach"]',
  REWARD_BUGS: '"green_cockroach"'
})
