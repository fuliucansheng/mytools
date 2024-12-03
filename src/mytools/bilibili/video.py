# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import fire
import re
import asyncio
import httpx
import logging
import pandas as pd
from bilibili_api import video, HEADERS

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %p",
    level="INFO",
)


def cleanup(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\r", " ", text)
    text = re.sub(r"\t", " ", text)

    return text


async def download_url(url: str, out: str):
    async with httpx.AsyncClient(headers=HEADERS) as sess:
        resp = await sess.get(url)
        length = resp.headers.get("content-length")
        with open(out, "wb") as f:
            process = 0
            for chunk in resp.iter_bytes(1024):
                if not chunk:
                    break

                process += len(chunk)
                f.write(chunk)


class Video:
    def download(
        self,
        bvid_file,
        output_folder="./",
        include_video=True,
    ):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        info = pd.read_csv(bvid_file, sep="\t", names=["bvid"], quoting=3)
        bvids = info.bvid.to_list()
        meta_infos = []
        for bvid in bvids:
            if bvid.strip() == "":
                continue
            bvideo = video.Video(bvid=bvid)
            bvideo_info = asyncio.run(bvideo.get_info())
            tname = bvideo_info["tname"]
            title = bvideo_info["title"]
            desc = bvideo_info["desc"]
            meta_infos.append(
                {
                    "bvid": bvideo_info["bvid"],
                    "tname": cleanup(tname if tname is not None else ""),
                    "title": cleanup(title if title is not None else ""),
                    "desc": cleanup(desc if desc is not None else ""),
                    "view": bvideo_info["stat"]["view"],
                    "like": bvideo_info["stat"]["like"],
                }
            )
            if include_video:
                download_url_data = asyncio.run(bvideo.get_download_url(0))
                detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
                streams = detecter.detect_best_streams()
                if detecter.check_flv_stream() == True:
                    asyncio.run(download_url(streams[0].url, "flv_temp.flv"))
                    os.system(
                        f"ffmpeg -loglevel quiet -i flv_temp.flv {output_folder}/{bvid}.mp4"
                    )
                    os.remove("flv_temp.flv")
                else:
                    asyncio.run(download_url(streams[0].url, "video_temp.m4s"))
                    asyncio.run(download_url(streams[1].url, "audio_temp.m4s"))
                    os.system(
                        f"ffmpeg -loglevel quiet -i video_temp.m4s -i audio_temp.m4s -vcodec copy -acodec copy -y {output_folder}/{bvid}.mp4"
                    )
                    os.remove("video_temp.m4s")
                    os.remove("audio_temp.m4s")
            logging.info(
                f"download video {bvid} | {bvideo_info['title']} | Include Video - {include_video} finish."
            )

        infos = pd.DataFrame()
        keys = meta_infos[0].keys()
        for key in keys:
            infos[key] = [_info[key] for _info in meta_infos]

        infos.to_csv(
            f"{output_folder}/metainfo.tsv",
            index=False,
            quoting=3,
            sep="\t",
        )


if __name__ == "__main__":
    fire.Fire(Video)
