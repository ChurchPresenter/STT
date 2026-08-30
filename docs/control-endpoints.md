# Driving STT from a button or another app

One press at the start of the preaching is worth more than any amount of detector tuning.
A mark says exactly where a phase began, from someone who was in the room, and it outranks
the detector everywhere downstream — the timeline, the sermon ranges and the summariser all
follow it. It is also evidence: the learner reads marks back afterwards, so the button that
fixes tonight's service is what stops next month's needing it.

This page is the contract. Anything that can send an HTTP request can use it: Bitfocus
Companion, a Stream Deck, a show-control cue, ChurchPresenter, or `curl` in a run sheet.

## The token

Control links carry the machine's access token in the URL:

```
?key=YOUR_ACCESS_TOKEN
```

Generate one in **Server Settings → Access token**. It does not expire, so a button
configured once keeps working. If no token is set, control links are refused rather than
left open — an installation that never opted in is not a control surface.

The token is required for the `GET` form specifically. A GET that changes something can be
triggered by any page a browser on the network happens to load, and that browser is already
authorised by its session cookie or by being on the whitelist; requiring the token means a
page that was never told it cannot fire these. `POST` requests go through the normal routes
and their normal gate.

> Not available over the Cloudflare tunnel — `?key=` is honoured on the local network only.

## Marking a phase

`GET` or `POST` `/api/control/phase-mark`

| Parameter | Meaning |
|---|---|
| `label` | The phase name, e.g. `Sermon 1`. Required unless `end` or `undo`. |
| `kind` | `S` speaking, `M` music, `_` quiet. Optional. |
| `at_ms` | Epoch milliseconds of the press. Defaults to when the server received it. |
| `end` | `1` to record that the phase ended, rather than starting one. |
| `undo` | `1` to remove the most recent mark. |
| `session` | An archived session id, to correct a past service. Omit for the live one. |
| `key` | The access token (see above). |

Copy-paste buttons:

```
http://STT-HOST:8080/api/control/phase-mark?key=TOKEN&label=Sermon%201&kind=S
http://STT-HOST:8080/api/control/phase-mark?key=TOKEN&label=Songs%202&kind=M
http://STT-HOST:8080/api/control/phase-mark?key=TOKEN&end=1
http://STT-HOST:8080/api/control/phase-mark?key=TOKEN&undo=1
```

The reply is JSON with a `text` field — one line suitable for a button's feedback, e.g.
`Marked Sermon 1`.

Send `at_ms` from the sending device's own clock if you can. A press travelling over a slow
link should land where the operator was, not where the server got to. Sending the same
`at_ms` twice is one decision, not two, so a button may be pressed again safely.

## Reading the state

`GET` `/api/service-phase` — the whole timeline: current phase, blocks, spans, marks and
corrections.

`GET` `/api/service-phase/profiles` — which service profile is in force, and **the phase
names this installation actually uses**. Read the label vocabulary from here rather than
hardcoding `Sermon`: a church that renamed its phases should see its own words on the
buttons.

`GET` `/api/transcription/status` — whether transcription is running.

Clients speaking socket.io also receive `service_phase_update` whenever the timeline
changes, which is the cheapest way to keep button feedback live.

## Starting and stopping

`POST` `/api/transcription/start` and `/api/transcription/stop`, with no body. These are
POST-only and are not exposed as control links: starting a service is not something a page
should be able to do by accident.

## Notes for ChurchPresenter

The useful integration is one call on the slide or section that begins the preaching, with
`at_ms` from ChurchPresenter's own clock. The phase names should come from
`/api/service-phase/profiles` rather than being fixed in the client, for the same reason the
detector's own thresholds are not shipped in the app: they belong to the church, not to the
software.

STT does not push. There is no outbound webhook and no callback — poll
`/api/service-phase`, or subscribe to `service_phase_update` over socket.io.
