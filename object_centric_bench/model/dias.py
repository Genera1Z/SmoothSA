from einops import rearrange
import torch as pt
import torch.nn as nn


class ARRandTransformerDecoder(nn.Module):
    """GeneralZ's new OCL decoder.
    Auto-regressive Transformer decoder with random token permutations.
    """

    def __init__(
        self,
        vfm_dim,
        posit_embed,
        # posit_embed_hw,
        project1,
        project2,
        backbone,
        readout,
    ):
        super().__init__()
        self.mask_token = nn.Parameter(pt.randn(1, 1, vfm_dim) * vfm_dim**-0.5)
        assert hasattr(posit_embed, "pe")
        self.posit_embed = posit_embed  # 1d
        # self.posit_embed_hw = posit_embed_hw  # 2d
        self.project1 = project1
        self.project2 = project2

        assert isinstance(backbone, nn.TransformerDecoder)
        self.norm0 = backbone.layers[0].norm1  # very beneficial
        backbone.layers[0].norm1 = nn.Identity()  # very beneficial
        self.backbone = backbone
        self.readout = readout

        def attent_hook_forward_pre(module, args, kwargs):
            kwargs["need_weights"] = True  # obtain the attention weights

        def attent_hook_forward(module, args, output):
            self._attent = output[1]

        self.backbone.layers[-1].multihead_attn.register_forward_pre_hook(
            attent_hook_forward_pre, with_kwargs=True
        )
        self.backbone.layers[-1].multihead_attn.register_forward_hook(
            attent_hook_forward
        )

        ### interaction asymmetry

        self._interact = [None for _ in range(len(self.backbone.layers[:-1]))]
        for l, layer in enumerate(self.backbone.layers[:-1]):

            def interact_hook_forward(module, args, output):
                self._interact[l] = output[1]

            layer.multihead_attn.register_forward_pre_hook(
                attent_hook_forward_pre, with_kwargs=True
            )
            layer.multihead_attn.register_forward_hook(interact_hook_forward)

    def forward(self, input, slots, p=0.5):
        """
        input: target to be destructed, shape=(b,m=h*w,c)
        slots: slots, shape=(b,n,c)
        """
        b, m, c = input.shape
        assert m == self.posit_embed.pe.size(1)
        _, n, _ = slots.shape
        device = input.device
        tokens = self.project1(input)  # (b,m,c)

        # TODO XXX disable masking in val for attent2 !!!

        # mim-predict-all-masked-tokens
        # seg1:
        # "ari": 0.20348355174064636, "ari_fg": 0.34435588121414185, "mbo": 0.29168349504470825, "miou": 0.2779198884963989
        # seg2:  # TODO disable masking in val for attent2 !!!
        # 'ari': 0.2038770616054535, 'ari_fg': 0.3444632291793823, 'mbo': 0.29167482256889343, 'miou': 0.27789679169654846

        if self.training:
            idxs = pt.vmap(  # (b,m)
                lambda _: pt.randperm(m, device=device), randomness="different"
            )(tokens)
            idxs_expanded = idxs[:, :, None].expand(-1, -1, c)

            idxs0 = pt.arange(0, m, device=device)[None, :]  # (1,m)
            keep1 = pt.randint(0, m - 1, [b, 1], device=device)  # (b,1)
            keep2 = (
                pt.ones(b, 1, dtype=pt.long, device=device) * int(256 * 0.1) - 1
            )  # TODO
            # TODO XXX realize a Poisson: when in [0, 1], it is Poisson; when out, then uniformly re-distribute in [0, 1]
            cond = pt.rand(b, 1, device=device) < p
            keep = pt.where(cond, keep1, keep2)
            # XXX 论文论述 XXX
            # keep@0: SlotMixerDecoder
            # 只预测下一个：ARTransformerDecoder
            # 其中9种只预测下一个，AR9TransformerDecoder
            mask = idxs0 < keep  # (b,m)

            # shuffle tokens
            tokens_shuffled = tokens.gather(1, idxs_expanded)  # (b,m,c)
            # mask tokens
            mask_token_expanded = self.mask_token.expand(b, m, -1)
            tokens_masked = tokens_shuffled.where(mask[:, :, None], mask_token_expanded)

            # shuffle pe
            pe_expanded = self.posit_embed.pe[:, :m, :].expand(b, -1, -1)  # (b,m,c)
            # pe_hw_expanded = self.posit_embed_hw.pe.flatten(1, -2)[:, :m, :].expand(
            #     b, -1, -1
            # )  # (b,m,c)
            pe_shuffled = pe_expanded.gather(1, idxs_expanded)  # (b,m,c)
            # pe_hw_shuffled = pe_hw_expanded.gather(1, idxs_expanded)  # (b,m,c)

            query = tokens_masked + pe_shuffled  # + pe_hw_shuffled

        else:
            query = (
                tokens
                + self.posit_embed.pe[:, :m, :]
                # + self.posit_embed_hw.pe.flatten(1, -2)[:, :m, :]
            )

        memory = self.project2(slots)
        autoreg = self.backbone(self.norm0(query), memory=memory)
        recon = self.readout(autoreg)  # (b,m,c)
        _, _, d = recon.shape

        if self.training:
            idxs_inverse = idxs.argsort(1)[:, :, None]
            recon = recon.gather(1, idxs_inverse.expand(-1, -1, d))
            attent = self._attent.gather(1, idxs_inverse.expand(-1, -1, n))
        else:
            attent = self._attent

        attent = rearrange(attent, "b m n -> b n m")
        return recon, attent
