def extractor(model, dataloader):
    def fliplr(img):
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    test_names = []
    test_features = torch.FloatTensor()

    for batch, sample in enumerate(tqdm(dataloader)):
        names, images = sample['name'], sample['img']

        ff = model(Variable(images.cuda(), volatile=True))[0].data.cpu()
        ff = ff + model(Variable(fliplr(images).cuda(), volatile=True))[0].data.cpu()
        ff = ff.div(torch.norm(ff, p=2, dim=1, keepdim=True).expand_as(ff))

        test_names = test_names + names
        test_features = torch.cat((test_features, ff), 0)

    return test_names, test_features