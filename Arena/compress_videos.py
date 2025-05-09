import logging
import imageio.v2 as iio
from pathlib import Path
from db_models import ORM, Experiment, Video, VideoPrediction, PoseEstimation
import config
import time


def get_videos_ids_for_compression(orm, sort_by_size=False):
    videos = {}
    with orm.session() as s:
        for v in s.query(Video).filter(Video.compression_status < 1).all():
            if 'tracking' in v.path:
                continue
            videos[v.id] = v.path
    # get videos sizes
    if not sort_by_size:
        return list(videos.keys())

    sizes = []
    for vid_id, vid_path in videos.items():
        try:
            size = Path(vid_path).stat().st_size
        except Exception:
            size = 0
        sizes.append((vid_id, size))

    videos_ids = [v for v, _ in sorted(sizes, key=lambda x: x[1], reverse=True)]
    return videos_ids


def compress(video_db_id, logger, orm):
    with orm.session() as s:
        v = s.query(Video).filter_by(id=video_db_id).first()
        assert v is not None, 'could not find video in DB'
        writer, reader = None, None
        source = Path(v.path).resolve()
        try:
            assert source.exists(), f'video does not exist'
            dest = compress_video_file(source, logger)
            v.path = str(dest)
            v.compression_status = 1
            source.unlink()

        except Exception as exc:
            v.compression_status = 2
            logger.error(f'Error compressing {source}; {exc}')

        finally:
            s.commit()
            if writer is not None:
                writer.close()
            if reader is not None:
                reader.close()
            time.sleep(2)


def compress_video_file(vid_path, logger=None):
    print_func = logger.info if logger is not None else print
    source = Path(vid_path)
    dest = source.with_suffix('.mp4')

    print_func(f'start video compression of {source}')
    t0 = time.time()
    reader = iio.get_reader(source.as_posix())
    fps = reader.get_meta_data()['fps']
    writer = iio.get_writer(dest.as_posix(), format="FFMPEG", mode="I",
                            fps=fps, codec="libx264", quality=5,
                            macro_block_size=8,  # to work with 1440x1080 image size
                            ffmpeg_log_level="error")
    for im in reader:
        writer.append_data(im)
    print_func(f'Finished compression of {dest} in {(time.time() - t0) / 60:.1f} minutes')
    return dest


def main():
    orm = ORM()
    with orm.session() as s:
        for v in s.query(Video).all():
            if v.compression_status or not (isinstance(v.path, str) and v.path.endswith('.avi')):
                continue


def foo():
    for p in Path('../output/experiments/PV26').rglob('*.avi'):
        if p.with_suffix('.mp4').exists():
            p.unlink()
            print(f'deleted {p}')


def clear_missing_videos():
    orm = ORM()
    with orm.session() as s:
        for v in s.query(Video).all():
            if not Path(v.path).exists():
                print(f'deleting from DB: {v.path}')
                for vp in s.query(VideoPrediction).filter_by(video_id=v.id).all():
                    s.delete(vp)
                for pe in s.query(PoseEstimation).filter_by(video_id=v.id).all():
                    pe.video_id = None
                s.delete(v)
                s.commit()


def compress_directory(dir_path, logger_):
    orm = ORM()
    with orm.session() as s:
        for p in Path(dir_path).rglob('*.avi'):
            vid = s.query(Video).filter_by(path=p.as_posix()).first()
            if vid is None:
                print(f'counld not find video in DB: {p}')
                continue
            compress(vid.id, logger_, orm)
            

def fix_wrong_vid_paths_in_db():
    orm = ORM()
    with orm.session() as s:
        for p in Path(config.EXPERIMENTS_DIR).rglob('*.avi'):
            vid = s.query(Video).filter_by(path=p.as_posix()).first()
            if vid is None:
                q = p.with_suffix('').as_posix()[:-2] + '%'
                vid = s.query(Video).filter(Video.path.ilike(q)).first()
                if vid is not None:
                    print(f'Videos name mismatch:\n{p.name} -> {Path(vid.path).name}')
                    vid.path = p.as_posix()
        s.commit()


if __name__ == "__main__":
    # fix_wrong_vid_paths_in_db()
    logger_ = logging.getLogger(__name__)
    logger_.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    logger_.addHandler(handler)
    