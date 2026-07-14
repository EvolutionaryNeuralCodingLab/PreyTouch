/* eslint-env jest */

import MirroredBug from './../../src/components/bugs/mirroredBug'

describe('mirroredBug circle start', () => {
  const circleTheta = MirroredBug.computed.circleTheta

  it('keeps the mirrored start angles for the first two bugs', () => {
    const leftTheta = circleTheta.call({isRightExit: false, bugId: 0})
    const rightTheta = circleTheta.call({isRightExit: true, bugId: 1})

    expect(leftTheta).toBeCloseTo((2 * Math.PI) / 3)
    expect(rightTheta).toBeCloseTo((Math.PI / 5) + 0.6)
  })
})
