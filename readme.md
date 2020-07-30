# Messenger Report Generator

For many of us, Messenger is the main communicator. It contains a lot of information about ourselves and our relationships. This repository contains a script that generates a bunch of charts about **your messages history**.

#### Charts this project generates:

- **messages count** rank
- **overall activity** over time
- average **activity over a day**
- average **activity over a week**
- average **message lengths** in _significant_ chats
- **word clouds** of important phrases in chats
- **activity over time** per chat
- **messages length distributions** in _significant_ chats
- **language diversity** rank _(experimental)_

## Usage

#### Collecting data

Facebook enables its users to get their Messenger **messages history**.

Data requesting steps:

1. Go to facebook settings and then proceed to [downloading your data](https://www.facebook.com/dyi/?referrer=yfi_settings).
1. Deselect all data and select only **Messages**
1. Choose data format to **JSON**
1. Choose the multimedia quality to **low** (all the media in chats are downloaded as well but they are omitted by the script)
1. Accept data request

Preparing data file shall not take more than 24h. You will be notified when your file is ready.

#### Setting up the script

After **cloning** this repository place the downloaded zip in `zips` directory and run:

```bash
pip install -r requirements.txt
python -m spacy download pl_core_news_md
python -m spacy download en_core_web_sm
```

In `params.json` you shall set your `"user"`, `"language"` and `"timezone"`.

```JSON
{
  "user": "Bartek Pogod",
  "language": "polish",
  "timezone": "Europe/Warsaw",

  [...]
}
```

#### Running script

If all is set up properly the charts shall be generated after running:

```bash
python messages_analysis.py
```

After a couple of minutes, all the plots shall appear in `figures` folder (or other specified in `params.json`).

## Contribute

The possibilities are almost endless. Take a look at the **issues** tab to write your own ideas or see how you can help! Let's make something great :D.
