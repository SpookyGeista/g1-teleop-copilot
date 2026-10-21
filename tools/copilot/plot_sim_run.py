#!/usr/bin/env python3
"""Plot an accepted recorded live-simulation run with auditable source hashes.

CPU-only; no model, simulator or robot transport is imported. Every plotted
point is a recorded sample. Undefined command/reference intervals remain gaps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def number(value):return type(value) in (int,float) and np.isfinite(value)
def comma(value,digits=3):return f'{value:.{digits}f}'.replace('.',',')


def final_preview(row):
    """Prefer the post-codec target check; retain legacy run compatibility."""
    actor=row.get('final_actor')
    if isinstance(actor,dict):
        if actor.get('applied') is not True:
            raise ValueError('Consumed command lacks an applied final actor')
        return actor['target_recheck']['preview_evidence']
    return row['handoff_target_recheck']['preview_evidence']


def build(directory,output):
    directory=Path(directory);output=Path(output)
    paths={name:directory/'simulation'/name for name in ('rows.json','report.json')}
    paths['phase_acceptance.json']=directory/'phase_acceptance.json'
    rows=json.loads(paths['rows.json'].read_text());report=json.loads(paths['report.json'].read_text())
    phase=json.loads(paths['phase_acceptance.json'].read_text())
    if (phase.get('accepted') is not True or report.get('live_ai') is not True or
            not phase.get('checks') or any(c.get('passed') is not True for c in phase['checks'])):
        raise ValueError('This figure requires an accepted live-simulation recording')
    if any(phase.get('artifact_sha256',{}).get(name)!=digest(paths[name]) for name in ('rows.json','report.json')):
        raise ValueError('Acceptance report refers to different recorded inputs')
    selected=[r for r in rows if type(r.get('scenario_tick')) is int and r['scenario_tick']>=0]
    x=np.array([r['scenario_tick'] for r in selected]);n=len(selected)
    if n<145 or not np.array_equal(x,np.arange(n)):
        raise ValueError('Complete indexed scenario with at least 145 steps required')
    consumed=np.array([r['command_consumed'] for r in selected],bool)
    angle=np.array([r['orientation_error_deg'] for r in selected],float)
    actual=np.array([r['actual_right_base_ee'] for r in selected],float)[:,:3,3]
    position=np.full(n,np.nan);step_time=np.full(n,np.nan);observation=np.full(n,np.nan)
    loop=np.array([r['timing']['loop_s'] for r in selected])*1000
    for i,row in enumerate(selected):
        c=row['controller'];age=c['snapshot'].get('input_age_s');stamp=c.get('operator_acquisition_timestamp')
        if consumed[i]:
            position[i]=np.linalg.norm(actual[i]-np.asarray(c['mapped_right_base_ee'])[:3,3])*1000
            e=final_preview(row)
            observation[i]=(e['checked_at']-e['preview_capture_timestamp'])*1000
        if number(age) and number(stamp):step_time[i]=stamp+age
    # Exact logged step(now) from source capture + controller input age. Only
    # adjacent recorded entries with both values are compared, never across HOLD.
    interval=np.full(n,np.nan)
    for i in range(1,n):
        if np.isfinite(step_time[i-1:i+1]).all():interval[i]=(step_time[i]-step_time[i-1])*1000
    excursion=np.linalg.norm(actual-actual[0],axis=1)*1000
    initial=float(np.median(angle[:4]));settled=float(np.median(angle[55:65]))
    maxima=dict(manual_position_error_mm=float(np.nanmax(position)),
        stationary_position_excursion_mm=float(excursion.max()),
        loop_ms=float(loop.max()),logged_adjacent_step_interval_ms=float(np.nanmax(interval)),
        handoff_observation_age_ms=float(np.nanmax(observation)))
    config=report['controller_configuration'];interval_budget=config['max_step_s']*1000
    position_budget=phase['thresholds']['maximum_manual_position_error_m']*1000
    excursion_budget=phase['thresholds']['maximum_stationary_position_excursion_m']*1000
    age_limits={final_preview(r)['max_preview_age_s']*1000
                for r in selected if r['command_consumed']}
    if len(age_limits)!=1:raise ValueError('Observation-age threshold changed during run')
    observation_budget=age_limits.pop()
    colors=dict(ink='#17212C',muted='#526174',teal='#007E80',blue='#3067A8',red='#A63346',
                gray='#A5ADB8',hold='#E6EAF0',stale='#F3C7CD',recover='#F8E4B5')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(14,9.8))
    fig.subplots_adjust(left=.075,right=.977,top=.79,bottom=.215,hspace=.40,wspace=.22)
    fig.text(.075,.962,'Live-KI-Teleop: strikte Simulationsabnahme bestanden',fontsize=18,
        fontweight='bold',color=colors['ink'])
    fig.text(.075,.925,f'G1 + Dex3 · gelernter Griff mit laufender Kameravalidierung · {n:,} Kontrollschritte · '
        f'{len(phase["checks"])}/{len(phase["checks"])} Prüfungen'.replace('1,000','1.000'),
        fontsize=11,color=colors['muted'])
    phase_handles=[Patch(facecolor=colors['hold'],label='HOLD / Clutch'),
        Patch(facecolor=colors['stale'],label='Tracking absichtlich veraltet: 94–100'),
        Patch(facecolor=colors['recover'],label='Quelle wieder da, weiter HOLD: 101–111'),
        Line2D([],[],color=colors['teal'],ls='--',lw=1.2,label='Bewusster Neu-Clutch: 112 / 116')]
    fig.legend(handles=phase_handles,loc='upper left',bbox_to_anchor=(.068,.898),ncol=2,
        frameon=False,fontsize=9.5,columnspacing=2.4,handlelength=2.1)
    for ax in axes.flat:
        ax.grid(axis='y',color='#E3E7ED',lw=.65,zorder=0)
        ax.set_xlabel('Kontrollschritt (keine Sekunden)')
    for ax in axes[0]:
        for a,b in ((0,4),(70,74)):ax.axvspan(a-.5,b+.5,color=colors['hold'],zorder=0)
        ax.axvspan(93.5,100.5,color=colors['stale'],zorder=0)
        ax.axvspan(100.5,111.5,color=colors['recover'],zorder=0)
        for tick in (112,116):ax.axvline(tick,color=colors['teal'],ls='--',lw=.75,alpha=.7,zorder=1)
        ax.set_xlim(-1,145);ax.set_xticks([0,25,50,75,100,125,145])
    ax=axes[0,0];visible=x<=145
    ax.scatter(x[visible&consumed],angle[visible&consumed],s=15,color=colors['teal'],zorder=3)
    ax.scatter(x[visible&~consumed],angle[visible&~consumed],s=15,color=colors['gray'],zorder=3)
    ax.set_ylim(-.5,max(19,initial*1.12));ax.set_ylabel('Tatsächlicher Winkelfehler rechts [°]')
    ax.set_title('A  Orientierung zum gewählten Griff',loc='left',pad=10)
    ax.text(.36,.89,f'{comma(initial,2)}° → {comma(settled)}°',transform=ax.transAxes,
        fontsize=14,fontweight='bold',color=colors['teal'])
    ax.text(.36,.78,'Median der Schritte 55–64\nFinaler Wert: '+comma(angle[-1])+'°',
        transform=ax.transAxes,fontsize=9.4,color=colors['muted'])
    ax.annotate('Assistenz loslassen',xy=(65,angle[65]),xytext=(44,7.1),fontsize=8.5,
        arrowprops=dict(arrowstyle='-',color=colors['muted'],lw=.7),color=colors['muted'])
    ax=axes[0,1]
    ax.scatter(x[visible],position[visible],s=15,color=colors['blue'],zorder=3)
    ax.axhline(position_budget,color=colors['red'],ls=(0,(4,3)),lw=1.2)
    ax.text(2,position_budget+.08,f'{comma(position_budget,0)} mm: Sollfehler-Budget',fontsize=9,color=colors['red'])
    ax.set_ylim(-.05,max(2.4,position_budget*1.2));ax.set_ylabel('TCP-Fehler zum manuellen Sollpunkt [mm]')
    ax.set_title('B  Position bleibt beim manuellen Soll',loc='left',pad=10)
    ax.text(.03,.69,'Maximum: '+comma(maxima['manual_position_error_mm'])+' mm',transform=ax.transAxes,
        fontsize=12,fontweight='bold',color=colors['blue'])
    ax.text(.03,.57,'Gesamtauslenkung: '+comma(maxima['stationary_position_excursion_mm'])+
        ' mm (Budget '+comma(excursion_budget,0)+' mm)',
        transform=ax.transAxes,fontsize=9.2,color=colors['muted'])
    ax=axes[1,0]
    ax.scatter(x,loop,s=5,color=colors['gray'],alpha=.65,label='Loopdauer',zorder=2)
    ax.scatter(x,interval,s=5,color=colors['teal'],alpha=.75,label='Abstand benachbarter step()-Aufrufe',zorder=3)
    ax.axhline(interval_budget,color=colors['red'],ls=(0,(4,3)),lw=1.2)
    ax.text(15,interval_budget+6,f'{comma(interval_budget,0)} ms: step()-Abstandsgrenze',fontsize=9,color=colors['red'])
    ax.set_title('C  Wallclock-Zeiten des Kontrollablaufs',loc='left',pad=10)
    ax.set_ylabel('Gemessene Dauer / Abstand [ms]');ax.set_ylim(0,max(240,interval_budget*1.2))
    ax.legend(loc='lower right',fontsize=8.4,frameon=False)
    ax=axes[1,1]
    ax.scatter(x,observation,s=6,color=colors['blue'],alpha=.8,zorder=3)
    ax.axhline(observation_budget,color=colors['red'],ls=(0,(4,3)),lw=1.2)
    ax.text(15,observation_budget+6,f'{comma(observation_budget,0)} ms: Beobachtungsalter-Grenze',fontsize=9,color=colors['red'])
    ax.set_title('D  Kamerabeobachtung bei letzter Freigabeprüfung',loc='left',pad=10)
    ax.set_ylabel('Capture → Handoff-Recheck [ms]');ax.set_ylim(0,max(295,observation_budget*1.18))
    ax.text(.035,.60,'Maximum: '+comma(maxima['handoff_observation_age_ms'],1)+' ms',
        transform=ax.transAxes,color=colors['blue'],fontsize=11,fontweight='bold')
    for ax in axes[1]:
        ax.set_xlim(0,n-1);ax.set_xticks([v for v in (0,200,400,600,800) if v<n-1]+[n-1])
    fig.text(.075,.145,'A/B: Ausschnitt 0–145. C/D und alle Maxima: gesamter Lauf. Punkte sind Einzelmessungen; keine Verbindung über HOLD.',fontsize=9.2,color=colors['muted'])
    fig.text(.075,.117,'Positionsfehler und Beobachtungsalter nur bei übernommenem Befehl; step()-Abstände nur zwischen benachbarten protokollierten Aufrufen.',fontsize=9.2,color=colors['muted'])
    phases=phase['metrics']['phase_commands'];stale=phases['stale'];restored=phases['restored_without_clutch']
    fig.text(.075,.089,f'{int(consumed.sum())} übernommene Befehle; Tracking-Ausfall {stale["consumed"]}/{stale["total"]} und '
        f'Quelle-zurück-ohne-Clutch {restored["consumed"]}/{restored["total"]} Befehle. Kontrollschritte sind keine Echtzeitachse.',
        fontsize=9.2,color=colors['ink'])
    fig.text(.075,.061,'Eine stationäre simulierte Szene, synthetische Eingabe; kein Griffschluss-/Hardware-/Nutzertest. Kontakte: '+
        comma(report['max_contact_force_n'],0)+' N im instrumentierten Bereich.',fontsize=9.2,color=colors['muted'])
    fig.text(.075,.033,'Quellenhashes, Grenzwerte und Messdefinitionen: '+output.with_suffix('.json').name,fontsize=8.8,color=colors['muted'])
    output=output.with_suffix('.png');output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=220,facecolor='white');plt.close(fig)
    payload=dict(schema='punktnull.live_acceptance_figure.v1',scope='single_accepted_scripted_live_simulation',
        source_sha256={str(p.resolve()):digest(p) for p in paths.values()},builder_sha256=digest(__file__),
        output_sha256={output.name:digest(output)},accepted=True,checks_passed=len(phase['checks']),
        scenario_steps=n,commands_consumed=int(consumed.sum()),initial_orientation_deg=initial,
        first_settled_orientation_deg=settled,final_orientation_deg=float(angle[-1]),maxima=maxima,
        budgets=dict(manual_position_error_mm=position_budget,stationary_excursion_mm=excursion_budget,step_interval_ms=interval_budget,
            observation_age_ms=observation_budget),
        plotted_sample_counts=dict(actual_orientation=n,manual_position_error=int(np.isfinite(position).sum()),
            loop_duration=n,adjacent_step_interval=int(np.isfinite(interval).sum()),handoff_observation_age=int(np.isfinite(observation).sum())),
        definitions=dict(orientation='Recorded actual post-physics right TCP error to frozen selected grasp orientation.',
            position='Post-physics right TCP position minus current consumed manual target; absent during HOLD.',
            excursion='Actual right TCP displacement from scenario step zero, including HOLD.',
            step_interval='Difference of (operator acquisition + controller input_age) for adjacent rows that recorded both values; exact controller step(now), no interpolation across missing entries.',
            loop='Recorded complete loop duration; distinct from the 200 ms step-interval bound.',
            observation='Actual last handoff-recheck time minus original camera capture timestamp, only for consumed commands.',
            x_axis='Control-step index only. Top panels show 0..145, bottom panels all scenario steps; no interpolated lines.'),
        limitations=['One scripted near-grasp orientation scenario with a stationary known simulated object.',
            'No human/headset or hardware trial and no grasp closure or lift.',
            'Contact telemetry covers 16 right arm/hand/wrist-camera bodies against table and target only.',
            'Measured wallclock durations and simulated physics time are distinct; no real-time guarantee or statistical success rate is inferred.'])
    output.with_suffix('.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return payload


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();payload=build(args.directory,args.output or args.directory/'live_acceptance.png')
    print(json.dumps(payload['maxima'],indent=2))


if __name__=='__main__':main()
