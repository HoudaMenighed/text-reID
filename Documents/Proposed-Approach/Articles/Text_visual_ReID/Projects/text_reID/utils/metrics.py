from prettytable import PrettyTable
import torch
import numpy as np
import logging
import torch.nn.functional as F


def rank_chunk(similarity, q_pids, g_pids, max_rank=10):
    """
    Evaluate a chunk of queries against the complete gallery.

    similarity:
        [num_queries_chunk, num_gallery]

    Returns sufficient statistics for accumulating:
        R@1, R@5, R@10, mAP, mINP
    """

    # We only need the complete ranking for mAP/mINP.
    indices = torch.argsort(
        similarity,
        dim=1,
        descending=True
    )

    # Move ranking indices to CPU before indexing CPU IDs.
    indices = indices.cpu()

    q_pids = q_pids.cpu()
    g_pids = g_pids.cpu()

    pred_labels = g_pids[indices]

    matches = pred_labels.eq(
        q_pids.view(-1, 1)
    )

    # CMC
    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1

    # Number of queries in this chunk
    num_queries = matches.size(0)

    # R@1/R@5/R@10 counts
    cmc_sum = all_cmc.float().sum(0)

    # mAP
    num_rel = matches.sum(1)

    tmp_cmc = matches.cumsum(1)

    precision = (
        tmp_cmc.float()
        / torch.arange(
            1,
            matches.size(1) + 1,
            dtype=torch.float32
        ).view(1, -1)
    )

    AP = (precision * matches.float()).sum(1) / num_rel.float()

    # mINP
    last_match = matches.nonzero(as_tuple=False)

    inp_values = []

    for row in range(num_queries):
        row_matches = matches[row].nonzero(as_tuple=False)

        if row_matches.numel() > 0:
            last_pos = row_matches[-1].item()

            inp = (
                tmp_cmc[row, last_pos].float()
                / (last_pos + 1.0)
            )

            inp_values.append(inp)

    if len(inp_values) > 0:
        inp_sum = torch.stack(inp_values).sum()
    else:
        inp_sum = torch.tensor(0.0)

    return (
        cmc_sum,
        AP.sum(),
        inp_sum,
        num_queries
    )


class Evaluator():

    def __init__(self, img_loader, txt_loader):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("IRRA.eval")

    @torch.no_grad()
    def _compute_embedding(self, model):

        model = model.eval()

        device = next(model.parameters()).device

        qids = []
        qfeats = []

        # -------------------------
        # Text embeddings
        # -------------------------
        for pid, caption in self.txt_loader:

            caption = caption.to(device)

            text_feat = model.encode_text(caption)

            # Immediately move feature to CPU.
            qfeats.append(
                F.normalize(text_feat, p=2, dim=1).cpu()
            )

            qids.append(
                pid.view(-1).cpu()
            )

        qids = torch.cat(qids, dim=0)
        qfeats = torch.cat(qfeats, dim=0)

        # -------------------------
        # Image embeddings
        # -------------------------
        gids = []
        gfeats = []

        for pid, img in self.img_loader:

            img = img.to(device)

            img_feat = model.encode_image(img)

            # Immediately move feature to CPU.
            gfeats.append(
                F.normalize(img_feat, p=2, dim=1).cpu()
            )

            gids.append(
                pid.view(-1).cpu()
            )

        gids = torch.cat(gids, dim=0)
        gfeats = torch.cat(gfeats, dim=0)

        return qfeats, gfeats, qids, gids

    @torch.no_grad()
    def _evaluate_direction(
        self,
        qfeats,
        gfeats,
        qids,
        gids,
        chunk_size=128
    ):

        num_queries = qfeats.size(0)

        cmc_sum = torch.zeros(10)
        ap_sum = torch.tensor(0.0)
        inp_sum = torch.tensor(0.0)

        total_queries = 0

        for start in range(
            0,
            num_queries,
            chunk_size
        ):

            end = min(
                start + chunk_size,
                num_queries
            )

            q_chunk = qfeats[start:end]

            # Only this chunk's similarity matrix exists.
            similarity = q_chunk @ gfeats.t()

            (
                chunk_cmc,
                chunk_ap,
                chunk_inp,
                chunk_n
            ) = rank_chunk(
                similarity=similarity,
                q_pids=qids[start:end],
                g_pids=gids,
                max_rank=10
            )

            cmc_sum += chunk_cmc

            ap_sum += chunk_ap

            inp_sum += chunk_inp

            total_queries += chunk_n

            # Explicitly release GPU/temporary tensors.
            del similarity
            del q_chunk

        cmc = (
            cmc_sum / total_queries
        ) * 100

        mAP = (
            ap_sum / total_queries
        ) * 100

        mINP = (
            inp_sum / total_queries
        ) * 100

        return cmc, mAP, mINP

    def eval(self, model, i2t_metric=False):

        qfeats, gfeats, qids, gids = \
            self._compute_embedding(model)

        # -------------------------
        # Text -> Image
        # -------------------------
        t2i_cmc, t2i_mAP, t2i_mINP = \
            self._evaluate_direction(
                qfeats=qfeats,
                gfeats=gfeats,
                qids=qids,
                gids=gids,
                chunk_size=128
            )

        t2i_cmc = t2i_cmc.numpy()
        t2i_mAP = t2i_mAP.item()
        t2i_mINP = t2i_mINP.item()

        table = PrettyTable(
            [
                "task",
                "R1",
                "R5",
                "R10",
                "mAP",
                "mINP"
            ]
        )

        table.add_row(
            [
                "t2i",
                t2i_cmc[0],
                t2i_cmc[4],
                t2i_cmc[9],
                t2i_mAP,
                t2i_mINP
            ]
        )

        # -------------------------
        # Image -> Text
        # -------------------------
        if i2t_metric:

            i2t_cmc, i2t_mAP, i2t_mINP = \
                self._evaluate_direction(
                    qfeats=gfeats,
                    gfeats=qfeats,
                    qids=gids,
                    gids=qids,
                    chunk_size=128
                )

            i2t_cmc = i2t_cmc.numpy()
            i2t_mAP = i2t_mAP.item()
            i2t_mINP = i2t_mINP.item()

            table.add_row(
                [
                    "i2t",
                    i2t_cmc[0],
                    i2t_cmc[4],
                    i2t_cmc[9],
                    i2t_mAP,
                    i2t_mINP
                ]
            )

        table.custom_format["R1"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["R5"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["R10"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["mAP"] = \
            lambda f, v: f"{v:.3f}"

        table.custom_format["mINP"] = \
            lambda f, v: f"{v:.3f}"

        self.logger.info(
            "\n" + str(table)
        )

        return t2i_cmc[0]