## Dataset collection notes

I want to learn how to create and curate a dataset for machine learning.

This part of the project focuses on searching for images, downloading them, validating them, and organizing them for training.

A single search term is usually not enough. For better dataset variety, each species should generate multiple related search queries.

Example:

`poison ivy` 
can be imporoved by searching for
- `poison ivy leaves`
- `plant`
- `at dark`
- `close up` 

This should produce more useful training data than relying on one broad search term.

Note: also searching for white oak alone returns a lot of house internal decor images.

## Notes

- Logging is important, we want to know what we imported and what failed. Need to be careful on not making this too noisy tho. Setting logging level can help with this.
- Don't fully trust the HTTP content type. Validate an image is actually an image (`from PIL import Image
`).
- Individual image failures should not stop the full import job.
- Candidate images should be manually reviewed before being moved into the trusted training dataset.

## TODO:

- Add retry logic with backoff on download.
- how does the image size effect the training step? should we set a minimum image size?
- blacklist site that adds trademark on top of the image.
    - alamy, dreamstime
- duplicate detection? can it be done by content and not just by name? is it worth it?
- parallel download 
