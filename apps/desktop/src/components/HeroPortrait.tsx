import { useEngineStore } from '../stores';
import type { HeroId } from '@shared';

interface HeroPortraitProps {
  heroId: HeroId;
  size?: number;
  showOrder?: boolean;
  order?: number;
  isBan?: boolean;
  className?: string;
}

export function HeroPortrait({ 
  heroId, 
  size = 48, 
  showOrder = false, 
  order,
  isBan = false,
  className = ''
}: HeroPortraitProps) {
  const getHero = useEngineStore((s) => s.getHero);
  const hero = getHero(heroId);
  const heroName = hero?.localized_name ?? `Hero ${heroId}`;
  const heroIcon = hero?.icon ?? '';

  return (
    <div 
      className={`relative flex-shrink-0 ${className}`}
      title={heroName}
    >
      <div 
        className="relative rounded-lg overflow-hidden border border-dota-border"
        style={{ width: size, height: size }}
      >
        {heroIcon ? (
          <img
            src={`https://cdn.cloudflare.steamstatic.com${heroIcon}`}
            alt={heroName}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <div className="w-full h-full bg-dota-border/50 flex items-center justify-center">
            <span className="text-xs text-dota-textMuted">{heroName.slice(0, 3)}</span>
          </div>
        )}
        
        {isBan && (
          <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
            <svg className="w-1/2 h-1/2 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
        )}

        {showOrder && order && (
          <div className="absolute -top-1 -right-1 w-5 h-5 bg-dota-accent text-white text-xs font-bold rounded-full flex items-center justify-center border-2 border-dota-bg">
            {order}
          </div>
        )}
      </div>
    </div>
  );
}