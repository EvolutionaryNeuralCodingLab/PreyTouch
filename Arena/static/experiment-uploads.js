function uploadCustomTrajectory(file) {
  const data = new FormData()
  data.append('file', file)

  return $.ajax({
    url: '/upload_custom_trajectory',
    type: 'POST',
    data: data,
    processData: false,
    contentType: false,
    dataType: 'json'
  })
}

function renderCustomTrajectories(blockId, trajectories) {
  const $list = $(`#customCsvList${blockId}`)

  $list.data('trajectories', trajectories)
  $list.empty()

  trajectories.forEach((trajectory, index) => {
    const $item = $('<li>').addClass('list-group-item')

    $('<div>')
      .text(trajectory.name)
      .appendTo($item)

    $('<small>')
      .addClass('d-block text-muted')
      .text(trajectory.path)
      .appendTo($item)

    $('<button>')
      .attr('type', 'button')
      .addClass('btn btn-sm btn-outline-danger mt-2')
      .text('Remove')
      .click(function () {
        const remaining = trajectories.filter((_, i) => i !== index)
        renderCustomTrajectories(blockId, remaining)
      })
      .appendTo($item)

    $list.append($item)
  })

  $(`#customTotal${blockId}`).text(trajectories.length)
}